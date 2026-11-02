"""
CPAP 분석 서버 v3 (무게중심 분석 전용)

흐름: 업로드(+키·윙스팬) → 포즈 탐지(extract_raw) → 보정·분석(refine, analyze) → 강습용 영상(render_lesson)

실행 (C:\\Project 에서, .venv 사용):
    .venv\\Scripts\\python server\\app.py               # http://localhost:8000
    .venv\\Scripts\\python server\\app.py --ngrok       # Vercel 웹에서 접속 (NGROK_AUTHTOKEN 필요)

설정: server/config.json (없으면 상위 폴더 params.json 의 pose_model_path, 그것도 없으면 yolo11l-pose.pt)
"""
import os
import sys
import json
import time
import uuid
import queue
import shutil
import argparse
import threading
import traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

VERSION = "3.0"
DATA_DIR = os.path.join(HERE, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
JOB_DIR = os.path.join(DATA_DIR, "jobs")
for d in (UPLOAD_DIR, OUTPUT_DIR, JOB_DIR):
    os.makedirs(d, exist_ok=True)

HEIGHT_RANGE = (100.0, 230.0)    # cm
WINGSPAN_RANGE = (100.0, 240.0)  # cm


# ────────────────────────────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────────────────────────────
def load_config():
    cfg = {
        "pose_model_path": "yolo11l-pose.pt",
        "pose_conf": 0.25,
        # Vercel 배포 주소와 로컬 개발 주소만 허용 (정규식)
        "allowed_origin_regex": r"https://.*\.vercel\.app|http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?",
        "web_url": "https://climbingposeanalyzer-local.vercel.app",   # 웹 화면 주소 (QR 코드에 사용)
        "ngrok_domain": "",   # ngrok 고정 도메인 (예: my-name.ngrok-free.app). 비우면 실행할 때마다 주소가 바뀜
        "ngrok_token": "",    # launcher.py 가 처음 실행 때 물어보고 저장
    }
    legacy = os.path.join(HERE, "..", "params.json")
    if os.path.exists(legacy):
        try:
            with open(legacy, encoding="utf-8") as f:
                p = json.load(f).get("pose_model_path")
            if p and os.path.exists(p):
                cfg["pose_model_path"] = p
        except Exception:
            pass
    path = os.path.join(HERE, "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


CFG = load_config()

app = FastAPI(title="CPAP 분석 서버", version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=CFG["allowed_origin_regex"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def ngrok_bypass(request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    # 크롬 로컬 네트워크 접근: HTTPS(Vercel) 페이지가 같은 PC의 localhost 서버를 부를 때 필요
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ────────────────────────────────────────────────────────────────────────────
# 작업 저장소 (메모리 + 파일. 서버를 재시작해도 결과가 남는다)
# ────────────────────────────────────────────────────────────────────────────
jobs: dict = {}
jobs_lock = threading.Lock()
job_queue: "queue.Queue[str]" = queue.Queue()


def _job_file(job_id):
    return os.path.join(JOB_DIR, f"{job_id}.json")


def save_job(job_id):
    with jobs_lock:
        data = dict(jobs[job_id])
    tmp = _job_file(job_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _job_file(job_id))


def update_job(job_id, **kw):
    with jobs_lock:
        jobs[job_id].update(kw)
    save_job(job_id)


def load_jobs():
    for fn in os.listdir(JOB_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(JOB_DIR, fn), encoding="utf-8") as f:
                j = json.load(f)
        except Exception:
            continue
        # 서버가 꺼질 때 진행 중이던 작업은 실패 처리
        if j.get("status") in ("queued", "processing"):
            j["status"] = "failed"
            j["error"] = "서버가 재시작되어 작업이 중단되었습니다. 다시 업로드해주세요."
        jobs[j["job_id"]] = j


load_jobs()


# ────────────────────────────────────────────────────────────────────────────
# 분석 작업 (한 번에 하나씩 처리)
# ────────────────────────────────────────────────────────────────────────────
STAGES = {"detect": (0, 70, "포즈 탐지"), "analyze": (70, 75, "보정·분석"), "render": (75, 100, "결과 영상 생성")}


def _progress_cb(job_id, stage):
    lo, hi, label = STAGES[stage]
    last = [0.0]

    def cb(done, total):
        now = time.time()
        if now - last[0] < 0.5 and done < total:
            return
        last[0] = now
        pct = lo + (hi - lo) * (done / max(total, 1))
        update_job(job_id, progress=round(pct, 1), stage=label, stage_detail=f"{done}/{total}")
    return cb


def run_job(job_id):
    import numpy as np
    import extract_raw
    import analyze as AN
    import render_lesson as RL

    j = jobs[job_id]
    try:
        update_job(job_id, status="processing", started_at=datetime.now().isoformat(timespec="seconds"),
                   progress=0, stage=STAGES["detect"][2])
        raw_path = os.path.join(OUTPUT_DIR, f"{j['name']}.raw.npz")
        extract_raw.extract(j["input"], raw_path, CFG["pose_model_path"], hold_model=None,
                            pose_conf=CFG["pose_conf"], progress=_progress_cb(job_id, "detect"))

        update_job(job_id, progress=STAGES["analyze"][0], stage=STAGES["analyze"][2], stage_detail="")
        A = AN.analyze(np.load(raw_path), j["height_cm"], j["wingspan_cm"])
        summary = RL.summarize(A)
        update_job(job_id, summary=summary)

        out_path = os.path.join(OUTPUT_DIR, f"{j['name']}_lesson.mp4")
        RL.render(A, j["input"], out_path, progress=_progress_cb(job_id, "render"))
        update_job(job_id, status="done", progress=100, stage="완료", stage_detail="",
                   output=out_path, finished_at=datetime.now().isoformat(timespec="seconds"))
    except Exception as e:
        traceback.print_exc()
        update_job(job_id, status="failed", error=f"{type(e).__name__}: {e}")


def worker():
    while True:
        job_id = job_queue.get()
        try:
            run_job(job_id)
        finally:
            job_queue.task_done()


threading.Thread(target=worker, daemon=True).start()


def _public(j):
    """웹에 돌려줄 필드만."""
    keys = ("job_id", "status", "progress", "stage", "stage_detail", "error", "climber_name",
            "height_cm", "wingspan_cm", "created_at", "finished_at", "original_filename")
    out = {k: j.get(k) for k in keys}
    if j.get("status") == "queued":
        with jobs_lock:
            waiting = [k for k, v in jobs.items() if v.get("status") == "queued" and v["created_at"] <= j["created_at"]]
        out["queue_position"] = len(waiting)
    return out


# ────────────────────────────────────────────────────────────────────────────
# API
# ────────────────────────────────────────────────────────────────────────────
_ngrok_url = ""
_ngrok_error = ""
_PORT = 8000


def _gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception:
        return ""


@app.get("/ping")
def ping():
    return {"status": "ok", "version": VERSION, "ngrok_url": _ngrok_url, "web_url": CFG["web_url"],
            "pose_model": os.path.basename(CFG["pose_model_path"]),
            "height_range": HEIGHT_RANGE, "wingspan_range": WINGSPAN_RANGE}


@app.get("/tunnel")
def tunnel():
    return {"ngrok_url": _ngrok_url, "active": bool(_ngrok_url), "error": _ngrok_error}


CONNECT_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CPAP 서버 연결</title>
<style>
 :root{--bg:#0c0d10;--s:#13151b;--s2:#1c1f28;--b:#252830;--f:#e8eaf0;--m:#8b93a1;--a:#22c55e;--w:#f59e0b}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--f);font:15px/1.6 system-ui,"Malgun Gothic",sans-serif}
 main{max-width:860px;margin:0 auto;padding:28px 16px}
 h1{font-size:22px;margin:0 0 4px} .sub{color:var(--m);margin:0 0 20px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px} @media(max-width:700px){.grid{grid-template-columns:1fr}}
 .card{background:var(--s);border:1px solid var(--b);border-radius:12px;padding:18px}
 .card h2{font-size:15px;margin:0 0 10px} .muted{color:var(--m);font-size:13px}
 #qr{background:#fff;padding:12px;border-radius:8px;display:inline-block;margin:6px 0 10px}
 a.btn{display:inline-block;background:var(--a);color:#06120a;font-weight:700;text-decoration:none;padding:10px 16px;border-radius:8px}
 code{background:var(--s2);border:1px solid var(--b);border-radius:6px;padding:2px 6px;font-size:12px;word-break:break-all}
 .row{display:flex;justify-content:space-between;gap:8px;padding:6px 0;border-bottom:1px solid var(--b);font-size:14px}
 .row:last-child{border:0} .ok{color:var(--a)} .warn{color:var(--w)}
</style></head><body><main>
<h1>CPAP 분석 서버 연결</h1>
<p class="sub">서버가 켜져 있는 동안 이 창은 닫아도 됩니다. 터미널 창을 닫으면 서버가 꺼집니다.</p>
<div class="grid">
 <div class="card">
  <h2>📱 휴대폰으로 연결</h2>
  <div id="mobile"></div>
 </div>
 <div class="card">
  <h2>💻 이 PC에서 열기</h2>
  <p class="muted">같은 PC에서는 ngrok 없이 바로 연결됩니다.</p>
  <a class="btn" id="pc" href="#">분석 웹 열기</a>
  <h2 style="margin-top:22px">서버 상태</h2>
  <div class="row"><span>버전</span><span>v__VERSION__</span></div>
  <div class="row"><span>GPU</span><span class="__GPU_CLS__">__GPU__</span></div>
  <div class="row"><span>포즈 모델</span><span>__MODEL__</span></div>
  <div class="row"><span>외부 접속(ngrok)</span><span class="__NG_CLS__">__NG_STATE__</span></div>
 </div>
</div>
<p class="muted" style="margin-top:18px">휴대폰 영상은 1080p로 찍으면 업로드가 빠릅니다. ngrok 무료 요금제는 월 전송량 한도가 있어 4K 영상은 피하세요.</p>
</main>
<template id="qrsvg">__QR_SVG__</template>
<script>
 const WEB="__WEB__", NGROK="__NGROK__", PORT="__PORT__";
 const pcUrl = WEB + "/?server=" + encodeURIComponent("http://localhost:" + PORT);
 document.getElementById("pc").href = pcUrl;
 const m = document.getElementById("mobile");
 if (NGROK) {
   const url = WEB + "/?server=" + encodeURIComponent(NGROK);
   m.innerHTML = '<p class="muted">휴대폰 카메라로 QR 코드를 찍으면 서버 주소가 자동으로 입력된 채로 열립니다.</p><div id="qr"></div><p class="muted">서버 주소: <code>' + NGROK + '</code></p>';
   const svg = document.getElementById("qrsvg").innerHTML.trim();
   if (svg) document.getElementById("qr").innerHTML = svg;
   else document.getElementById("qr").outerHTML = '<p><a href="' + url + '">' + url + '</a></p>';
 } else {
   m.innerHTML = '<p class="warn">ngrok이 연결되지 않아 휴대폰에서는 접속할 수 없습니다.</p><p class="muted">__NG_ERR__</p><p class="muted">CPAP 서버 시작 프로그램을 다시 실행해 ngrok 토큰을 입력하세요.</p>';
 }
</script></body></html>"""


def _qr_svg(text, size=220):
    """OpenCV로 QR 코드를 만들어 SVG 문자열로 (인터넷·추가 패키지 없이 동작)."""
    try:
        import cv2
        import numpy as np
        img = cv2.QRCodeEncoder.create().encode(text)      # 모듈 1칸 = 1px, 여백 포함
        if img is None or img.size == 0:
            return ""
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        n = img.shape[0]
        dark = np.argwhere(img < 128)
        rects = "".join(f'<rect x="{x}" y="{y}" width="1" height="1"/>' for y, x in dark)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" width="{size}" height="{size}" '
                f'shape-rendering="crispEdges"><rect width="{n}" height="{n}" fill="#fff"/><g fill="#000">{rects}</g></svg>')
    except Exception:
        return ""


@app.get("/connect")
def connect_page():
    from fastapi.responses import HTMLResponse
    import html as _h
    from urllib.parse import quote as _quote
    gpu = _gpu_name()
    page = (CONNECT_HTML
            .replace("__VERSION__", VERSION)
            .replace("__GPU_CLS__", "ok" if gpu else "warn").replace("__GPU__", _h.escape(gpu or "없음 (CPU로 느리게 동작)"))
            .replace("__MODEL__", _h.escape(os.path.basename(CFG["pose_model_path"])))
            .replace("__NG_CLS__", "ok" if _ngrok_url else "warn")
            .replace("__NG_STATE__", "연결됨" if _ngrok_url else "꺼짐")
            .replace("__NG_ERR__", _h.escape(_ngrok_error))
            .replace("__WEB__", CFG["web_url"].rstrip("/"))
            .replace("__NGROK__", _ngrok_url)
            .replace("__QR_SVG__", _qr_svg(f"{CFG['web_url'].rstrip('/')}/?server={_quote(_ngrok_url, safe='')}") if _ngrok_url else "")
            .replace("__PORT__", str(_PORT)))
    return HTMLResponse(page)


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    height_cm: float = Form(...),
    wingspan_cm: float = Form(...),
    climber_name: str = Form(""),
):
    if not (HEIGHT_RANGE[0] <= height_cm <= HEIGHT_RANGE[1]):
        raise HTTPException(422, f"키는 {HEIGHT_RANGE[0]:.0f}~{HEIGHT_RANGE[1]:.0f}cm 범위여야 합니다.")
    if not (WINGSPAN_RANGE[0] <= wingspan_cm <= WINGSPAN_RANGE[1]):
        raise HTTPException(422, f"윙스팬은 {WINGSPAN_RANGE[0]:.0f}~{WINGSPAN_RANGE[1]:.0f}cm 범위여야 합니다.")

    job_id = str(uuid.uuid4())
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id[:4]}"
    ext = os.path.splitext(file.filename or "")[1].lower() or ".mp4"
    input_path = os.path.join(UPLOAD_DIR, name + ext)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id, "name": name, "status": "queued", "progress": 0, "stage": "대기 중",
            "stage_detail": "", "error": "", "input": input_path, "output": "",
            "original_filename": file.filename, "climber_name": climber_name.strip()[:40],
            "height_cm": height_cm, "wingspan_cm": wingspan_cm,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    save_job(job_id)
    job_queue.put(job_id)
    return {"job_id": job_id}


def _get(job_id):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return j


@app.get("/status/{job_id}")
def status(job_id: str):
    return _public(_get(job_id))


@app.get("/summary/{job_id}")
def summary(job_id: str):
    j = _get(job_id)
    if not j.get("summary"):
        raise HTTPException(404, "아직 분석 요약이 없습니다.")
    return {**_public(j), "summary": j["summary"]}


@app.get("/video/{job_id}")
def video(job_id: str):
    j = _get(job_id)
    if j.get("status") != "done" or not os.path.exists(j.get("output", "")):
        raise HTTPException(404, "결과 영상이 아직 없습니다.")
    return FileResponse(j["output"], media_type="video/mp4", filename=f"{j['name']}_lesson.mp4")


@app.get("/jobs")
def list_jobs(limit: int = 50):
    with jobs_lock:
        items = sorted(jobs.values(), key=lambda v: v.get("created_at", ""), reverse=True)[:limit]
    return {"jobs": [_public(j) for j in items]}


@app.get("/videos")
def list_videos():
    """영상 비교 화면용: 완료된 결과 영상 목록."""
    with jobs_lock:
        done = [j for j in jobs.values() if j.get("status") == "done" and os.path.exists(j.get("output", ""))]
    done.sort(key=lambda v: v.get("created_at", ""), reverse=True)
    vids = []
    for j in done:
        label = " · ".join(x for x in (j.get("climber_name"), j.get("created_at", "").replace("T", " ")[:16]) if x)
        vids.append({"name": label or os.path.basename(j["output"]), "url": f"/outputs/{os.path.basename(j['output'])}",
                     "job_id": j["job_id"]})
    return {"videos": vids}


# ────────────────────────────────────────────────────────────────────────────
# ngrok
# ────────────────────────────────────────────────────────────────────────────
def _find_ngrok_token(cli_token=""):
    """토큰 찾는 순서: --token → NGROK_AUTHTOKEN 환경변수 → ngrok 프로그램 설정 파일(ngrok config add-authtoken)."""
    if cli_token:
        return cli_token, "--token"
    if os.environ.get("NGROK_AUTHTOKEN"):
        return os.environ["NGROK_AUTHTOKEN"], "NGROK_AUTHTOKEN 환경변수"
    if CFG.get("ngrok_token"):
        return CFG["ngrok_token"], "server/config.json"
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(os.environ.get("LOCALAPPDATA", os.path.join(home, "AppData", "Local")), "ngrok", "ngrok.yml"),
        os.path.join(home, "AppData", "Roaming", "ngrok", "ngrok.yml"),
        os.path.join(home, ".config", "ngrok", "ngrok.yml"),
        os.path.join(home, "Library", "Application Support", "ngrok", "ngrok.yml"),
        os.path.join(home, ".ngrok2", "ngrok.yml"),
    ]
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("authtoken:"):
                        tok = line.split(":", 1)[1].strip().strip("'\"")
                        if tok:
                            return tok, path
        except OSError:
            continue
    return "", ""


def _start_ngrok(port, token=""):
    global _ngrok_url, _ngrok_error
    try:
        import ngrok as ngrok_sdk
        opts = {"authtoken": token} if token else {"authtoken_from_env": True}
        domain = (CFG.get("ngrok_domain") or "").strip().replace("https://", "").rstrip("/")
        try:
            listener = ngrok_sdk.forward(port, domain=domain, **opts) if domain else ngrok_sdk.forward(port, **opts)
        except Exception as e:
            if not domain:
                raise
            print(f"[ngrok] 고정 도메인({domain}) 연결 실패: {e} → 임시 주소로 다시 시도합니다.")
            listener = ngrok_sdk.forward(port, **opts)
        _ngrok_url = listener.url()
        _ngrok_error = ""
        print(f"\n{'=' * 60}\n  ngrok 공개 URL: {_ngrok_url}\n  웹 상단 '로컬 서버' 입력란에 이 주소를 넣으세요.\n{'=' * 60}\n")
    except ImportError:
        _ngrok_error = "ngrok 패키지가 없습니다 (pip install ngrok)"
        print("[ERROR] " + _ngrok_error)
    except Exception as e:
        _ngrok_error = f"ngrok 시작 실패: {e}"
        print(f"[ERROR] ngrok 시작 실패: {e}")
        if "4018" in str(e) or "authenticated" in str(e):
            print("  → ngrok 토큰이 없거나 잘못되었습니다. 아래 중 하나를 실행한 뒤 다시 시작하세요.")
            print("     1) 이번 창에서만:  set NGROK_AUTHTOKEN=토큰")
            print("     2) 영구 저장:      setx NGROK_AUTHTOKEN 토큰   (새 cmd 창에서 적용)")
            print("     토큰 확인: https://dashboard.ngrok.com/get-started/your-authtoken")
        print("  (서버는 http://localhost 로는 계속 사용할 수 있습니다)")


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser(description="CPAP 분석 서버")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--ngrok", action="store_true")
    ap.add_argument("--token", default="")
    a = ap.parse_args()
    _PORT = a.port
    if a.ngrok:
        tok, src = _find_ngrok_token(a.token)
        if tok:
            print(f"[ngrok] 토큰 사용: {src}")
            threading.Thread(target=_start_ngrok, args=(a.port, tok), daemon=True).start()
            time.sleep(2)
        else:
            _ngrok_error = "ngrok 토큰이 설정되지 않았습니다."
            print("[ngrok] 토큰을 찾지 못해 ngrok 없이 시작합니다.")
            print("  → set NGROK_AUTHTOKEN=토큰  (또는 setx 로 영구 저장) 후 다시 실행하세요.")
            print("     토큰 확인: https://dashboard.ngrok.com/get-started/your-authtoken")
    print(f"\nCPAP 분석 서버 v{VERSION}: http://localhost:{a.port}")
    print(f"포즈 모델: {CFG['pose_model_path']}\n")
    uvicorn.run(app, host=a.host, port=a.port, reload=False)
