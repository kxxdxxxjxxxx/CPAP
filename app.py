import os
import sys
import uuid
import json
import shutil
import zipfile
import threading
import argparse
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline import process_video

app = FastAPI(title="Climbing Pose Analyzer API")

# ────────────────────────────────────────────────────────────────────────────
# ngrok 터널 상태 (런타임 공유)
# ────────────────────────────────────────────────────────────────────────────
_ngrok_url: str = ""

# CORS - 웹 퍼블리싱(Vercel 등)에서 로컬 서버로 요청 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ngrok 무료 플랜 브라우저 경고 페이지 우회:
# 응답 헤더에 ngrok-skip-browser-warning 을 반영해 클라이언트 요청을 통과시킴
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

class NgrokBypassMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        response.headers["ngrok-skip-browser-warning"] = "true"
        return response

app.add_middleware(NgrokBypassMiddleware)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
TRAIN_DIR = "train_data"
LABEL_DIR = "label_sessions"
POSE_DIR = "pose_sessions"
TEMPLATES_DIR = "templates"
PARAMS_FILE = "params.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(POSE_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/label_files", StaticFiles(directory=LABEL_DIR), name="label_files")
app.mount("/pose_files", StaticFiles(directory=POSE_DIR), name="pose_files")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ────────────────────────────────────────────────────────────────────────────
# 기본 파라미터 (웹에서 튜닝 가능)
# ────────────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "occlusion_threshold": 0.3,
    "occlusion_search_radius": 150,
    "smoothing_alpha": 0.5,
    "hold_conf": 0.1,
    "pose_conf": 0.35,
    "arm_extension": 0.15,
    "leg_extension": 0.10,
    "hold_model_path": "C:/Project/runs/detect/train11/weights/best.pt",
    # 손목 hold-snap 을 몸에 가려진 홀드로만 한정 (잘 보이는 홀드로의 잘못된 점프 방지)
    "snap_only_body_occluded": True,
    "body_occlusion_overlap_threshold": 0.4,
}

def load_params():
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, "r") as f:
            saved = json.load(f)
        return {**DEFAULT_PARAMS, **saved}
    return DEFAULT_PARAMS.copy()

def save_params(params: dict):
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

# ────────────────────────────────────────────────────────────────────────────
# 작업(Job) 관리 - 영상 분석 + 학습 공통
# ────────────────────────────────────────────────────────────────────────────
jobs: dict = {}  # job_id -> {status, output, error, progress, type}
train_jobs: dict = {}  # train_job_id -> {status, error, progress, log}

# ────────────────────────────────────────────────────────────────────────────
# 기존 HTML 인터페이스 (하위 호환용)
# ────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

# ────────────────────────────────────────────────────────────────────────────
# 헬스 체크 (웹에서 서버 연결 확인용)
# ────────────────────────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    params = load_params()
    return JSONResponse({
        "status": "ok",
        "version": "2.0",
        "params": params,
        "ngrok_url": _ngrok_url,
    })

# ────────────────────────────────────────────────────────────────────────────
# ngrok 터널 정보 조회
# ────────────────────────────────────────────────────────────────────────────
@app.get("/tunnel")
def get_tunnel():
    return JSONResponse({
        "ngrok_url": _ngrok_url,
        "active": bool(_ngrok_url),
    })

# ────────────────────────────────────────────────────────────────────────────
# 파라미터 조회 / 저장
# ────────────────────────────────────────────────────────────────────────────
@app.get("/params")
def get_params():
    return JSONResponse(load_params())

class ParamsUpdate(BaseModel):
    occlusion_threshold: float | None = None
    occlusion_search_radius: float | None = None
    smoothing_alpha: float | None = None
    hold_conf: float | None = None
    pose_conf: float | None = None
    arm_extension: float | None = None
    leg_extension: float | None = None
    hold_model_path: str | None = None
    # Phase 1 신규 안정화 파라미터
    one_euro_mincutoff: float | None = None
    one_euro_beta: float | None = None
    max_speed_ratio: float | None = None
    max_predict_frames: int | None = None
    wrist_hold_snap_ratio: float | None = None
    pose_model_path: str | None = None
    # 손목 hold-snap 을 몸에 가려진 홀드로만 한정
    snap_only_body_occluded: bool | None = None
    body_occlusion_overlap_threshold: float | None = None

@app.post("/params")
def update_params(body: ParamsUpdate):
    current = load_params()
    updated = body.model_dump(exclude_none=True)
    current.update(updated)
    save_params(current)
    # pipeline 모듈에 파라미터 실시간 반영
    _apply_params_to_pipeline(current)
    return JSONResponse({"status": "saved", "params": current})


class ActivateModelBody(BaseModel):
    path: str
    kind: str  # "hold" | "pose"


@app.post("/params/activate-model")
def activate_model(body: ActivateModelBody):
    """학습 후 best.pt 경로를 즉시 활성 모델로 적용 (params.json 갱신 + pipeline 핫스왑).
    학습 페이지의 '활성 모델로 적용' 버튼이 호출한다.
    """
    norm = body.path.replace("\\", "/")
    if not os.path.exists(norm):
        raise HTTPException(status_code=400, detail=f"파일이 존재하지 않습니다: {norm}")
    if body.kind not in ("hold", "pose"):
        raise HTTPException(status_code=400, detail="kind must be 'hold' or 'pose'")
    cur = load_params()
    key = "hold_model_path" if body.kind == "hold" else "pose_model_path"
    cur[key] = norm
    save_params(cur)
    _apply_params_to_pipeline(cur)
    return JSONResponse({"status": "activated", "kind": body.kind, "path": norm})

def _apply_params_to_pipeline(params: dict):
    """파라미터를 pipeline.py 모듈에 동적으로 적용"""
    try:
        import pipeline
        import cog
        pipeline.OCCLUSION_THRESHOLD = params.get("occlusion_threshold", 0.3)
        pipeline.OCCLUSION_SEARCH_RADIUS = params.get("occlusion_search_radius", 150)
        pipeline.SMOOTHING_ALPHA = params.get("smoothing_alpha", 0.5)
        pipeline.HOLD_CONF = params.get("hold_conf", 0.1)
        pipeline.POSE_CONF = params.get("pose_conf", 0.35)
        pipeline.ARM_EXTENSION = params.get("arm_extension", 0.15)
        pipeline.LEG_EXTENSION = params.get("leg_extension", 0.10)
        # Phase 1 신규 안정화 파라미터
        pipeline.ONE_EURO_MINCUTOFF = params.get("one_euro_mincutoff", 1.0)
        pipeline.ONE_EURO_BETA = params.get("one_euro_beta", 0.05)
        pipeline.MAX_SPEED_RATIO = params.get("max_speed_ratio", 0.12)
        pipeline.MAX_PREDICT_FRAMES = int(params.get("max_predict_frames", 10))
        pipeline.WRIST_HOLD_SNAP_RATIO = params.get("wrist_hold_snap_ratio", 0.04)
        # 손목 hold-snap "몸 가림 한정" 모드
        pipeline.SNAP_ONLY_BODY_OCCLUDED = bool(params.get("snap_only_body_occluded", True))
        pipeline.BODY_OCCLUSION_OVERLAP_THRESHOLD = float(params.get("body_occlusion_overlap_threshold", 0.4))
        # 홀드 모델 경로가 변경됐으면 재로드
        new_path = params.get("hold_model_path", "")
        if new_path and os.path.exists(new_path):
            from ultralytics import YOLO
            pipeline.hold_model = YOLO(new_path)
            print(f"[params] hold_model 로드: {new_path}")
        # 포즈 모델 경로도 갱신 가능
        new_pose = params.get("pose_model_path", "")
        if new_pose and os.path.exists(new_pose):
            from ultralytics import YOLO
            pipeline.pose_model = YOLO(new_pose)
            print(f"[params] pose_model 로드: {new_pose}")
    except Exception as e:
        print(f"[params] pipeline 적용 실패: {e}")


# ────────────────────────────────────────────────────────────────────────────
# 서버 시작 시 params.json 의 모델 경로/파라미터를 즉시 pipeline 에 반영
# (기존 버그: pipeline.py 가 import 시 하드코딩 경로만 로드해서, 재시작 후
#  포즈/홀드 학습 결과가 자동 적용되지 않았음)
# ────────────────────────────────────────────────────────────────────────────
_apply_params_to_pipeline(load_params())

# ────────────────────────────────────────────────────────────────────────────
# 영상 분석
# ────────────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    display_option: str = Form("no_ears")
):
    job_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{job_id[:4]}"
    input_path = os.path.join(UPLOAD_DIR, f"{filename}.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"{filename}_out.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    jobs[job_id] = {
        "status": "queued",
        "output": output_path,
        "error": "",
        "type": "analysis",
    }
    background_tasks.add_task(run_job, job_id, input_path, output_path, display_option)
    return JSONResponse({"job_id": job_id})

def run_job(job_id, input_path, output_path, display_option):
    jobs[job_id]["status"] = "processing"
    params = load_params()
    _apply_params_to_pipeline(params)
    try:
        process_video(input_path, output_path, display_option)
        jobs[job_id]["status"] = "done"
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "not found"}, status_code=404)
    return jobs[job_id]

@app.get("/result/{job_id}")
def result(job_id: str):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        return JSONResponse({"error": "not ready"}, status_code=404)
    return JSONResponse({"output_url": f"/video/{job_id}"})

@app.get("/video/{job_id}")
def video(job_id: str):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        return JSONResponse({"error": "not ready"}, status_code=404)
    output_path = jobs[job_id]["output"]
    return FileResponse(output_path, media_type="video/mp4", filename=f"{job_id}_out.mp4")

@app.get("/videos")
def list_videos():
    files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".mp4")]
    files.sort(reverse=True)
    return JSONResponse({
        "videos": [
            {"name": f, "url": f"/outputs/{f}"}
            for f in files
        ]
    })

@app.post("/videos/reencode")
def reencode_outputs():
    """outputs 폴더의 모든 mp4를 브라우저 호환 H.264로 재인코딩.
    이미 H.264인 파일도 안전하게 재인코딩한다 (멱등).
    """
    from pipeline import _transcode_to_web_h264, _get_ffmpeg_exe
    if _get_ffmpeg_exe() is None:
        raise HTTPException(status_code=500, detail="ffmpeg을 찾을 수 없습니다. 'pip install imageio-ffmpeg' 후 서버를 재시작하세요.")

    files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".mp4")]
    converted, failed = [], []
    for fn in files:
        src = os.path.join(OUTPUT_DIR, fn)
        tmp = os.path.join(OUTPUT_DIR, f".__reenc_{fn}")
        try:
            shutil.move(src, tmp)
            _transcode_to_web_h264(tmp, src)
            if not os.path.exists(src):
                # 재인코딩 실패 시 원본 복구
                if os.path.exists(tmp):
                    shutil.move(tmp, src)
                failed.append(fn)
            else:
                converted.append(fn)
        except Exception as e:
            failed.append(f"{fn}: {e}")
            if os.path.exists(tmp) and not os.path.exists(src):
                shutil.move(tmp, src)
    return JSONResponse({"converted": converted, "failed": failed})

# ────────────────────────────────────────────────────────────────────────────
# 홀드 탐지 학습
# ────────────────��───────────────────────────────────────────────────────────
@app.post("/train/upload")
async def upload_train_data(files: list[UploadFile] = File(...)):
    """이미지 + YOLO 라��(txt) 파일 업로드"""
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(TRAIN_DIR, session_id)
    images_dir = os.path.join(session_dir, "images")
    labels_dir = os.path.join(session_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    uploaded = []
    for upload in files:
        ext = os.path.splitext(upload.filename)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            dest = os.path.join(images_dir, upload.filename)
        elif ext == ".txt":
            dest = os.path.join(labels_dir, upload.filename)
        else:
            continue
        content = await upload.read()
        with open(dest, "wb") as f:
            f.write(content)
        uploaded.append(upload.filename)

    return JSONResponse({"session_id": session_id, "uploaded": uploaded})

@app.post("/train/start/{session_id}")
def start_training(session_id: str, epochs: int = 50, imgsz: int = 640):
    """지정 세션의 데이터로 YOLOv11 학습 시작"""
    session_dir = os.path.join(TRAIN_DIR, session_id)
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="session not found")

    train_job_id = str(uuid.uuid4())
    train_jobs[train_job_id] = {
        "status": "queued",
        "error": "",
        "progress": 0,
        "log": [],
        "session_id": session_id,
        "best_model": None,
    }
    thread = threading.Thread(
        target=_run_training,
        args=(train_job_id, session_dir, epochs, imgsz),
        daemon=True
    )
    thread.start()
    return JSONResponse({"train_job_id": train_job_id})

def _run_training(train_job_id: str, session_dir: str, epochs: int, imgsz: int):
    from ultralytics import YOLO
    import yaml

    train_jobs[train_job_id]["status"] = "processing"
    images_dir = os.path.join(session_dir, "images")
    labels_dir = os.path.join(session_dir, "labels")

    # 간단한 train/val 분할 (80/20)
    img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    split = max(1, int(len(img_files) * 0.8))
    train_imgs = img_files[:split]
    val_imgs = img_files[split:]

    train_img_dir = os.path.join(session_dir, "train", "images")
    train_lbl_dir = os.path.join(session_dir, "train", "labels")
    val_img_dir = os.path.join(session_dir, "val", "images")
    val_lbl_dir = os.path.join(session_dir, "val", "labels")
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    for fn in train_imgs:
        shutil.copy2(os.path.join(images_dir, fn), os.path.join(train_img_dir, fn))
        lbl = os.path.splitext(fn)[0] + ".txt"
        src_lbl = os.path.join(labels_dir, lbl)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, os.path.join(train_lbl_dir, lbl))

    for fn in val_imgs:
        shutil.copy2(os.path.join(images_dir, fn), os.path.join(val_img_dir, fn))
        lbl = os.path.splitext(fn)[0] + ".txt"
        src_lbl = os.path.join(labels_dir, lbl)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, os.path.join(val_lbl_dir, lbl))

    # data.yaml 생성
    data_yaml = {
        "path": os.path.abspath(session_dir),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["hold"],
    }
    yaml_path = os.path.join(session_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, allow_unicode=True)

    try:
        model = YOLO("yolo11n.pt")

        # 커스텀 콜백으로 진행률 추적
        total_epochs = epochs
        def on_train_epoch_end(trainer):
            ep = trainer.epoch + 1
            pct = int(ep / total_epochs * 100)
            train_jobs[train_job_id]["progress"] = pct
            metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
            log_line = f"Epoch {ep}/{total_epochs} | progress: {pct}%"
            if metrics:
                for k, v in metrics.items():
                    log_line += f" | {k}: {v:.4f}" if isinstance(v, float) else f" | {k}: {v}"
            train_jobs[train_job_id]["log"].append(log_line)

        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        result = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            workers=0,
            project=os.path.abspath(os.path.join(session_dir, "runs")).replace("\\", "/"),
            name="train",
            exist_ok=True,
        )
        best_path = str(result.save_dir / "weights" / "best.pt").replace("\\", "/") if result else ""
        train_jobs[train_job_id]["status"] = "done"
        train_jobs[train_job_id]["progress"] = 100
        train_jobs[train_job_id]["best_model"] = best_path
        train_jobs[train_job_id]["log"].append(f"학습 완료. 모델 저장: {best_path}")
    except Exception as e:
        train_jobs[train_job_id]["status"] = "failed"
        train_jobs[train_job_id]["error"] = str(e)
        train_jobs[train_job_id]["log"].append(f"오류: {e}")

@app.get("/train/status/{train_job_id}")
def train_status(train_job_id: str):
    if train_job_id not in train_jobs:
        return JSONResponse({"error": "not found"}, status_code=404)
    return train_jobs[train_job_id]

@app.get("/train/download/{train_job_id}")
def download_model(train_job_id: str):
    """학습 완료된 best.pt 다운로드"""
    if train_job_id not in train_jobs:
        raise HTTPException(status_code=404, detail="job not found")
    job = train_jobs[train_job_id]
    if job["status"] != "done" or not job.get("best_model"):
        raise HTTPException(status_code=400, detail="model not ready")
    return FileResponse(job["best_model"], filename="best.pt", media_type="application/octet-stream")

@app.get("/train/sessions")
def list_train_sessions():
    if not os.path.exists(TRAIN_DIR):
        return JSONResponse({"sessions": []})
    sessions = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    result = []
    for s in sessions:
        img_count = 0
        idir = os.path.join(TRAIN_DIR, s, "images")
        if os.path.exists(idir):
            img_count = len([f for f in os.listdir(idir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        result.append({"session_id": s, "image_count": img_count})
    return JSONResponse({"sessions": result})


# ────────────────────────────────────────────────────────────────────────────
# 클릭 라벨링 - 벽 사진을 업로드하고 클릭으로 라벨을 만들어 학습
# ────────────────────────────────────────────────────────────────────────────
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _label_session_dir(session_id: str) -> str:
    return os.path.join(LABEL_DIR, session_id)


def _ensure_label_session(session_id: str) -> tuple[str, str, str]:
    base = _label_session_dir(session_id)
    if not os.path.exists(base):
        raise HTTPException(status_code=404, detail="label session not found")
    return base, os.path.join(base, "images"), os.path.join(base, "labels")


def _read_image_size(path: str) -> tuple[int, int]:
    """이미지 width/height을 읽음. cv2 사용."""
    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            return 0, 0
        h, w = img.shape[:2]
        return int(w), int(h)
    except Exception:
        return 0, 0


def _count_label_boxes(label_path: str) -> int:
    if not os.path.exists(label_path):
        return 0
    try:
        with open(label_path, "r") as f:
            return sum(1 for line in f if line.strip() and not line.strip().startswith("#"))
    except Exception:
        return 0


_FS_FORBIDDEN = set('<>:"/\\|?*')

def _safe_session_id(raw: str, fallback_prefix: str = "session") -> str:
    """파일시스템 금지문자만 제거하고 입력값 그대로 사용 (한글 OK).
    Ultralytics 가 비-ASCII 경로를 깨뜨리는 문제는 학습 시작 시
    _migrate_to_ascii_session() 으로 ASCII 사본 폴더를 자동 생성해서 회피한다.
    """
    sid = "".join(c for c in raw.strip() if c not in _FS_FORBIDDEN and not c.isspace())
    if not sid:
        sid = datetime.now().strftime(f"{fallback_prefix}_%Y%m%d_%H%M%S")
    return sid


@app.post("/label/sessions")
def create_label_session(name: str = Form("")):
    """새 라벨링 세션 생성. 입력한 이름을 그대로 폴더명으로 사용 (한글 가능)."""
    raw = name.strip()
    sid = _safe_session_id(raw, "session")
    base = _label_session_dir(sid)
    if os.path.exists(base) and raw:
        sid = f"{sid}_{uuid.uuid4().hex[:4]}"
        base = _label_session_dir(sid)
    os.makedirs(os.path.join(base, "images"), exist_ok=True)
    os.makedirs(os.path.join(base, "labels"), exist_ok=True)
    return JSONResponse({"session_id": sid, "original_name": raw})


@app.get("/label/sessions")
def list_label_sessions():
    if not os.path.exists(LABEL_DIR):
        return JSONResponse({"sessions": []})
    sessions = []
    for d in sorted(os.listdir(LABEL_DIR), reverse=True):
        full = os.path.join(LABEL_DIR, d)
        if not os.path.isdir(full):
            continue
        idir = os.path.join(full, "images")
        ldir = os.path.join(full, "labels")
        img_count = 0
        labeled = 0
        total_boxes = 0
        if os.path.exists(idir):
            for fn in os.listdir(idir):
                if fn.lower().endswith(IMG_EXTS):
                    img_count += 1
                    lbl = os.path.join(ldir, os.path.splitext(fn)[0] + ".txt")
                    n = _count_label_boxes(lbl)
                    if n > 0:
                        labeled += 1
                        total_boxes += n
        sessions.append({
            "session_id": d,
            "image_count": img_count,
            "labeled_count": labeled,
            "total_boxes": total_boxes,
        })
    return JSONResponse({"sessions": sessions})


@app.delete("/label/sessions/{session_id}")
def delete_label_session(session_id: str):
    base = _label_session_dir(session_id)
    if not os.path.exists(base):
        raise HTTPException(status_code=404, detail="not found")
    shutil.rmtree(base)
    return JSONResponse({"status": "deleted"})


@app.post("/label/upload/{session_id}")
async def upload_label_images(session_id: str, files: list[UploadFile] = File(...)):
    """벽 사진 다중 업로드 (라벨은 비어있는 상태로 시작)"""
    base, images_dir, labels_dir = _ensure_label_session(session_id)
    saved = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in IMG_EXTS:
            continue
        # 파일명 충돌 방지
        safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in f.filename)
        dest = os.path.join(images_dir, safe_name)
        if os.path.exists(dest):
            stem, e = os.path.splitext(safe_name)
            safe_name = f"{stem}_{uuid.uuid4().hex[:6]}{e}"
            dest = os.path.join(images_dir, safe_name)
        content = await f.read()
        with open(dest, "wb") as fh:
            fh.write(content)
        w, h = _read_image_size(dest)
        saved.append({"filename": safe_name, "width": w, "height": h})
    return JSONResponse({"uploaded": saved})


@app.get("/label/images/{session_id}")
def list_label_images_in_session(session_id: str):
    """세션 내 이미지 목록 + 각 이미지의 라벨 박스 수, 크기"""
    base, images_dir, labels_dir = _ensure_label_session(session_id)
    items = []
    for fn in sorted(os.listdir(images_dir)):
        if not fn.lower().endswith(IMG_EXTS):
            continue
        full = os.path.join(images_dir, fn)
        w, h = _read_image_size(full)
        lbl = os.path.join(labels_dir, os.path.splitext(fn)[0] + ".txt")
        items.append({
            "filename": fn,
            "width": w,
            "height": h,
            "label_count": _count_label_boxes(lbl),
            "url": f"/label_files/{session_id}/images/{fn}",
        })
    return JSONResponse({"images": items})


@app.delete("/label/image/{session_id}/{filename}")
def delete_label_image(session_id: str, filename: str):
    base, images_dir, labels_dir = _ensure_label_session(session_id)
    img_path = os.path.join(images_dir, filename)
    lbl_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
    if os.path.exists(img_path):
        os.remove(img_path)
    if os.path.exists(lbl_path):
        os.remove(lbl_path)
    return JSONResponse({"status": "deleted"})


class LabelBox(BaseModel):
    cx: float  # 0..1
    cy: float  # 0..1
    w: float   # 0..1
    h: float   # 0..1


class SaveLabelsBody(BaseModel):
    boxes: list[LabelBox]


@app.post("/label/labels/{session_id}/{filename}")
def save_labels(session_id: str, filename: str, body: SaveLabelsBody):
    """YOLO 형식으로 라벨 저장 (class_id=0 hold)"""
    base, images_dir, labels_dir = _ensure_label_session(session_id)
    img_path = os.path.join(images_dir, filename)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="image not found")
    lbl_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
    lines = []
    for b in body.boxes:
        cx = max(0.0, min(1.0, b.cx))
        cy = max(0.0, min(1.0, b.cy))
        w = max(0.001, min(1.0, b.w))
        h = max(0.001, min(1.0, b.h))
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(lbl_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return JSONResponse({"status": "saved", "count": len(lines)})


@app.get("/label/labels/{session_id}/{filename}")
def get_labels(session_id: str, filename: str):
    base, images_dir, labels_dir = _ensure_label_session(session_id)
    lbl_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
    boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        boxes.append({
                            "cx": float(parts[1]),
                            "cy": float(parts[2]),
                            "w": float(parts[3]),
                            "h": float(parts[4]),
                        })
                    except ValueError:
                        continue
    return JSONResponse({"boxes": boxes})


def _gather_existing_dataset(existing_path: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    기존 데이터셋 경로에서 (image_path, label_path) 페어를 train/val로 반환.
    지원 구조:
      1) <root>/train/images, <root>/train/labels, <root>/val/images, <root>/val/labels
      2) <root>/images, <root>/labels (평탄)
    이미지/라벨 매칭은 stem(확장자 제외) 기준.
    """
    if not existing_path or not os.path.exists(existing_path):
        return [], []

    def collect(img_dir: str, lbl_dir: str) -> list[tuple[str, str]]:
        if not os.path.exists(img_dir):
            return []
        items = []
        for fn in os.listdir(img_dir):
            if not fn.lower().endswith(IMG_EXTS):
                continue
            stem = os.path.splitext(fn)[0]
            lbl = os.path.join(lbl_dir, stem + ".txt")
            if os.path.exists(lbl):
                items.append((os.path.join(img_dir, fn), lbl))
        return items

    # train/val 구조 시도
    tr = collect(os.path.join(existing_path, "train", "images"), os.path.join(existing_path, "train", "labels"))
    va = collect(os.path.join(existing_path, "val", "images"), os.path.join(existing_path, "val", "labels"))
    if tr or va:
        return tr, va

    # 평탄 구조
    flat = collect(os.path.join(existing_path, "images"), os.path.join(existing_path, "labels"))
    if flat:
        n = max(1, int(len(flat) * 0.8))
        return flat[:n], flat[n:]

    return [], []


def _run_label_training(
    train_job_id: str,
    session_id: str,
    epochs: int,
    imgsz: int,
    combine_with_existing: bool,
    existing_dataset_path: str,
    initial_weights: str = "",
    auto_activate: bool = False,
):
    from ultralytics import YOLO
    import yaml
    from pathlib import Path as _Path

    train_jobs[train_job_id]["status"] = "processing"
    base = _label_session_dir(session_id)
    src_images = os.path.join(base, "images")
    src_labels = os.path.join(base, "labels")

    # 학습용 임시 디렉터리 (매번 새로)
    work = os.path.join(base, "_work")
    if os.path.exists(work):
        shutil.rmtree(work)
    train_img = os.path.join(work, "train", "images")
    train_lbl = os.path.join(work, "train", "labels")
    val_img = os.path.join(work, "val", "images")
    val_lbl = os.path.join(work, "val", "labels")
    for d in [train_img, train_lbl, val_img, val_lbl]:
        os.makedirs(d, exist_ok=True)

    log = train_jobs[train_job_id]["log"]

    # 1. 이번 세션에서 라벨이 있는 이미지만 수집
    own_pairs: list[tuple[str, str]] = []
    for fn in sorted(os.listdir(src_images)):
        if not fn.lower().endswith(IMG_EXTS):
            continue
        stem = os.path.splitext(fn)[0]
        lbl = os.path.join(src_labels, stem + ".txt")
        if os.path.exists(lbl) and _count_label_boxes(lbl) > 0:
            own_pairs.append((os.path.join(src_images, fn), lbl))
    log.append(f"라벨링된 이미지 {len(own_pairs)}장 수집")

    if not own_pairs and not combine_with_existing:
        train_jobs[train_job_id]["status"] = "failed"
        train_jobs[train_job_id]["error"] = "라벨링된 이미지가 없습니다."
        return

    # 80/20 분할
    split = max(1, int(len(own_pairs) * 0.8))
    own_train = own_pairs[:split]
    own_val = own_pairs[split:] if len(own_pairs) > 1 else own_pairs[:0]

    # 2. 기존 데이터셋 결합
    ext_train: list[tuple[str, str]] = []
    ext_val: list[tuple[str, str]] = []
    if combine_with_existing:
        ext_train, ext_val = _gather_existing_dataset(existing_dataset_path)
        log.append(f"기존 데이터셋 결합: train {len(ext_train)}장, val {len(ext_val)}장 ({existing_dataset_path or '경로 없음'})")
        if not ext_train and not ext_val:
            log.append("경���: 기존 데이터셋을 찾지 못했습니다. 새 라벨로만 학���합니다.")

    # 3. work 폴더로 복사
    def copy_pair(pair, dst_img_dir, dst_lbl_dir, prefix=""):
        img_src, lbl_src = pair
        fn = (prefix + os.path.basename(img_src))
        shutil.copy2(img_src, os.path.join(dst_img_dir, fn))
        new_lbl = os.path.splitext(fn)[0] + ".txt"
        shutil.copy2(lbl_src, os.path.join(dst_lbl_dir, new_lbl))

    for p in own_train:
        copy_pair(p, train_img, train_lbl)
    for p in own_val:
        copy_pair(p, val_img, val_lbl)
    for p in ext_train:
        copy_pair(p, train_img, train_lbl, prefix="ext_")
    for p in ext_val:
        copy_pair(p, val_img, val_lbl, prefix="ext_")

    # val이 비어있으면 train 한 장 옮기기 (YOLO val 필수)
    if not os.listdir(val_img) and os.listdir(train_img):
        # 첫 train 한 쌍을 val로 이동
        for fn in os.listdir(train_img):
            shutil.move(os.path.join(train_img, fn), os.path.join(val_img, fn))
            stem = os.path.splitext(fn)[0]
            lbl = stem + ".txt"
            if os.path.exists(os.path.join(train_lbl, lbl)):
                shutil.move(os.path.join(train_lbl, lbl), os.path.join(val_lbl, lbl))
            break

    n_train = len([f for f in os.listdir(train_img) if f.lower().endswith(IMG_EXTS)])
    n_val = len([f for f in os.listdir(val_img) if f.lower().endswith(IMG_EXTS)])
    log.append(f"최종 학습셋: train={n_train}, val={n_val}")

    # 4. data.yaml
    # Ultralytics는 yaml의 path: 키를 자기 글로벌 datasets_dir(C:\Project\datasets)
    # 기준으로 재해석하는 경향이 있다. 이를 회피하기 위해:
    #   - path: 키를 yaml에서 제거 (그러면 Ultralytics가 yaml 파일이 있는 폴더를 path로 사용)
    #   - train/val 은 절대 POSIX 경로로 직접 기록
    work_abs = os.path.abspath(work).replace("\\", "/")
    train_img_abs = os.path.abspath(train_img).replace("\\", "/")
    val_img_abs = os.path.abspath(val_img).replace("\\", "/")
    data_yaml = {
        "train": train_img_abs,
        "val": val_img_abs,
        "nc": 1,
        "names": ["hold"],
    }
    yaml_path = os.path.abspath(os.path.join(work, "data.yaml")).replace("\\", "/")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, allow_unicode=True)
    log.append(f"data.yaml: {yaml_path}")
    log.append(f"  train: {train_img_abs} ({n_train}장)")
    log.append(f"  val:   {val_img_abs} ({n_val}장)")

    # 학습 직전 경로 사전 검증 - 디렉터리/이미지 수 부족 시 즉시 오류
    if not os.path.isdir(val_img_abs):
        train_jobs[train_job_id]["status"] = "failed"
        train_jobs[train_job_id]["error"] = f"val 이미지 폴더가 없습니다: {val_img_abs}"
        return
    val_images_actual = [f for f in os.listdir(val_img_abs) if f.lower().endswith(IMG_EXTS)]
    if len(val_images_actual) == 0:
        train_jobs[train_job_id]["status"] = "failed"
        train_jobs[train_job_id]["error"] = f"val 이미지가 0장입니다: {val_img_abs}"
        return

    # Ultralytics 글로벌 DATASETS_DIR 모듈 상수를 work_abs 로 패치
    # (SETTINGS.update 만으로는 import 시점에 캡처된 모듈 상수 DATASETS_DIR 가 갱신되지 않음)
    try:
        import ultralytics.utils as _ul_utils
        import ultralytics.data.utils as _ul_data_utils
        from ultralytics.utils import SETTINGS as _ULSETTINGS
        _ULSETTINGS.update({"datasets_dir": work_abs})
        _ul_utils.DATASETS_DIR = _Path(work_abs)
        _ul_data_utils.DATASETS_DIR = _Path(work_abs)
        log.append(f"DATASETS_DIR 패치: {work_abs}")
    except Exception as _e:
        log.append(f"warning: DATASETS_DIR 패치 실패 ({_e}) - 계속 진행")

    # 5. 학습
    try:
        # 사용자가 초기 가중치를 지정했으면 해당 best.pt에서 fine-tuning 시작
        weights_to_load = "yolo11n.pt"
        if initial_weights:
            iw = initial_weights.replace("\\", "/")
            if os.path.exists(iw):
                weights_to_load = iw
                log.append(f"초기 가중치 로드: {weights_to_load} (fine-tuning)")
            else:
                log.append(f"경고: 초기 가중치 파일을 찾지 못했습니다 ({iw}). 기본 yolo11n.pt 사용")
        else:
            log.append("초기 가중치: yolo11n.pt (사전학습 모델)")
        model = YOLO(weights_to_load)

        total_epochs = epochs
        def on_train_epoch_end(trainer):
            ep = trainer.epoch + 1
            pct = int(ep / total_epochs * 100)
            train_jobs[train_job_id]["progress"] = pct
            line = f"Epoch {ep}/{total_epochs} | progress: {pct}%"
            metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
            if metrics:
                for k, v in metrics.items():
                    line += f" | {k}: {v:.4f}" if isinstance(v, float) else f" | {k}: {v}"
            train_jobs[train_job_id]["log"].append(line)

        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        # project 를 절대경로로 지정해야 Ultralytics 가 자체 'runs/detect/' 를 prepend 하지 않는다.
        result = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            workers=0,
            project=os.path.abspath(os.path.join(work, "runs")).replace("\\", "/"),
            name="train",
            exist_ok=True,
        )
        best_path = str(result.save_dir / "weights" / "best.pt").replace("\\", "/") if result else ""
        train_jobs[train_job_id]["status"] = "done"
        train_jobs[train_job_id]["progress"] = 100
        train_jobs[train_job_id]["best_model"] = best_path
        train_jobs[train_job_id]["log"].append(f"학습 완료. 모델 저장: {best_path}")

        # 자동 활성화: params.json 의 hold_model_path 갱신 + pipeline 핫스왑
        if auto_activate:
            if best_path and os.path.exists(best_path):
                try:
                    cur = load_params()
                    cur["hold_model_path"] = best_path
                    save_params(cur)
                    _apply_params_to_pipeline(cur)
                    train_jobs[train_job_id]["log"].append(
                        "활성 홀드 모델로 자동 적용됨 (params.json 갱신, 재시작 후에도 유지)"
                    )
                    train_jobs[train_job_id]["activated"] = True
                except Exception as ee:
                    train_jobs[train_job_id]["log"].append(f"자동 활성화 실패: {ee}")
            else:
                train_jobs[train_job_id]["log"].append(
                    f"자동 활성화 건너뜀 - best.pt 파일 확인 실패 ({best_path})"
                )
    except Exception as e:
        train_jobs[train_job_id]["status"] = "failed"
        train_jobs[train_job_id]["error"] = str(e)
        train_jobs[train_job_id]["log"].append(f"오류: {e}")


def _ascii_safe(s: str) -> bool:
    return all(c.isascii() and (c.isalnum() or c in "-_") for c in s)


def _migrate_to_ascii_session(session_id: str) -> str:
    """세션 ID가 비-ASCII면 ASCII 전용 ID로 폴더를 복사해 새 세션을 만든다.
    Ultralytics 가 학습 중 경로의 비-ASCII 문자를 내부에서 잘라내 데이터셋을 못 찾는 문제를 회피.
    이미 ASCII이면 그대로 반환.
    """
    if _ascii_safe(session_id):
        return session_id
    new_sid = "".join(c for c in session_id if c.isascii() and (c.isalnum() or c in "-_"))
    if not new_sid:
        new_sid = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    # 충돌 방지
    if os.path.exists(_label_session_dir(new_sid)):
        new_sid = f"{new_sid}_{uuid.uuid4().hex[:4]}"
    src = _label_session_dir(session_id)
    dst = _label_session_dir(new_sid)
    shutil.copytree(src, dst)
    return new_sid


@app.post("/label/train/{session_id}")
def start_label_training(
    session_id: str,
    epochs: int = 50,
    imgsz: int = 640,
    combine_with_existing: bool = True,
    existing_dataset_path: str = "",
    initial_weights: str = "",
    auto_activate: bool = False,
):
    """클릭 라벨링 데이터로 학습 시작 (옵션: 기존 데이터셋 결합 / 기존 best.pt fine-tuning).
    세션 ID가 비-ASCII(한글 등) 면 자동으로 ASCII 전용 폴더로 복사 후 학습 (Ultralytics 호환).
    auto_activate=True 면 학습 완료 시 best.pt를 활성 홀드 모델로 자동 설정.
    """
    base = _label_session_dir(session_id)
    if not os.path.exists(base):
        raise HTTPException(status_code=404, detail="label session not found")

    # 비-ASCII 세션이면 ASCII 전용 폴더로 자동 복사 (Ultralytics가 한글 경로를 깨뜨리는 버그 회피)
    effective_session = _migrate_to_ascii_session(session_id)

    train_job_id = str(uuid.uuid4())
    train_jobs[train_job_id] = {
        "status": "queued",
        "error": "",
        "progress": 0,
        "log": (
            [f"세션 ID에 비-ASCII 문자가 포함되어 ASCII 전용 사본 '{effective_session}' 으로 학습합니다."]
            if effective_session != session_id else []
        ),
        "session_id": effective_session,
        "original_session_id": session_id,
        "best_model": None,
        "activated": False,
    }
    thread = threading.Thread(
        target=_run_label_training,
        args=(train_job_id, effective_session, epochs, imgsz, combine_with_existing, existing_dataset_path, initial_weights, auto_activate),
        daemon=True,
    )
    thread.start()
    return JSONResponse({"train_job_id": train_job_id, "effective_session_id": effective_session})


# ════════════════════════════════════════════════════════════════════════════
# Phase 2: 포즈 키포인트 라벨링 - 클라이밍 자세 데이터 수집 + 파인튜닝
# ════════════════════════════════════════════════════════════════════════════
# COCO-17 키포인트 좌우 대칭 인덱스 (수평 뒤집기 augmentation 시 좌우 키포인트 교환)
COCO_KPT_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def _pose_session_dir(session_id: str) -> str:
    return os.path.join(POSE_DIR, session_id)


def _ensure_pose_session(session_id: str) -> tuple[str, str, str]:
    base = _pose_session_dir(session_id)
    if not os.path.exists(base):
        raise HTTPException(status_code=404, detail="pose session not found")
    return base, os.path.join(base, "images"), os.path.join(base, "labels")


def _count_pose_label(label_path: str) -> int:
    """라벨 파일이 존재하고 1개 이상의 포즈 라인이 있으면 1, 아니면 0"""
    if not os.path.exists(label_path):
        return 0
    try:
        with open(label_path, "r") as f:
            for line in f:
                if line.strip() and not line.strip().startswith("#"):
                    return 1
        return 0
    except Exception:
        return 0


@app.post("/pose/sessions")
def create_pose_session(name: str = Form("")):
    """포즈 라벨링 세션 생성. 입력한 이름을 그대로 폴더명으로 사용 (한글 가능).
    학습 시 Ultralytics 호환을 위해 ASCII 사본이 자동 생성된다."""
    raw = name.strip()
    sid = _safe_session_id(raw, "pose")
    base = _pose_session_dir(sid)
    if os.path.exists(base) and raw:
        sid = f"{sid}_{uuid.uuid4().hex[:4]}"
        base = _pose_session_dir(sid)
    os.makedirs(os.path.join(base, "images"), exist_ok=True)
    os.makedirs(os.path.join(base, "labels"), exist_ok=True)
    return JSONResponse({"session_id": sid, "original_name": raw})


@app.get("/pose/sessions")
def list_pose_sessions():
    if not os.path.exists(POSE_DIR):
        return JSONResponse({"sessions": []})
    sessions = []
    for d in sorted(os.listdir(POSE_DIR), reverse=True):
        full = os.path.join(POSE_DIR, d)
        if not os.path.isdir(full):
            continue
        idir = os.path.join(full, "images")
        ldir = os.path.join(full, "labels")
        img_count = labeled = 0
        if os.path.exists(idir):
            for fn in os.listdir(idir):
                if fn.lower().endswith(IMG_EXTS):
                    img_count += 1
                    lbl = os.path.join(ldir, os.path.splitext(fn)[0] + ".txt")
                    if _count_pose_label(lbl):
                        labeled += 1
        sessions.append({
            "session_id": d,
            "image_count": img_count,
            "labeled_count": labeled,
        })
    return JSONResponse({"sessions": sessions})


@app.delete("/pose/sessions/{session_id}")
def delete_pose_session(session_id: str):
    base = _pose_session_dir(session_id)
    if not os.path.exists(base):
        raise HTTPException(status_code=404, detail="not found")
    shutil.rmtree(base)
    return JSONResponse({"status": "deleted"})


@app.post("/pose/upload/{session_id}")
async def upload_pose_images(session_id: str, files: list[UploadFile] = File(...)):
    base, images_dir, labels_dir = _ensure_pose_session(session_id)
    saved = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in IMG_EXTS:
            continue
        safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in f.filename)
        dest = os.path.join(images_dir, safe_name)
        if os.path.exists(dest):
            stem, e = os.path.splitext(safe_name)
            safe_name = f"{stem}_{uuid.uuid4().hex[:6]}{e}"
            dest = os.path.join(images_dir, safe_name)
        content = await f.read()
        with open(dest, "wb") as fh:
            fh.write(content)
        w, h = _read_image_size(dest)
        saved.append({"filename": safe_name, "width": w, "height": h})
    return JSONResponse({"uploaded": saved})


@app.get("/pose/images/{session_id}")
def list_pose_images(session_id: str):
    base, images_dir, labels_dir = _ensure_pose_session(session_id)
    items = []
    for fn in sorted(os.listdir(images_dir)):
        if not fn.lower().endswith(IMG_EXTS):
            continue
        full = os.path.join(images_dir, fn)
        w, h = _read_image_size(full)
        lbl = os.path.join(labels_dir, os.path.splitext(fn)[0] + ".txt")
        items.append({
            "filename": fn,
            "width": w,
            "height": h,
            "labeled": bool(_count_pose_label(lbl)),
            "url": f"/pose_files/{session_id}/images/{fn}",
        })
    return JSONResponse({"images": items})


@app.delete("/pose/image/{session_id}/{filename}")
def delete_pose_image(session_id: str, filename: str):
    base, images_dir, labels_dir = _ensure_pose_session(session_id)
    img_path = os.path.join(images_dir, filename)
    lbl_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
    if os.path.exists(img_path):
        os.remove(img_path)
    if os.path.exists(lbl_path):
        os.remove(lbl_path)
    return JSONResponse({"status": "deleted"})


@app.post("/pose/predict/{session_id}/{filename}")
def predict_pose_bootstrap(session_id: str, filename: str):
    """현재 yolo11l-pose 모델로 17 키포인트를 예측해 라벨링 부트스트랩 제공.

    Returns: {"keypoints": [{"x": 0..1, "y": 0..1, "v": 0|1|2}, ... 17개],
              "found": bool, "image": {"width": w, "height": h}}
    검출 실패 시 keypoints 는 17개 모두 (0.5, 0.5, 0) 으로 채워서 반환.
    """
    base, images_dir, _ = _ensure_pose_session(session_id)
    img_path = os.path.join(images_dir, filename)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="image not found")

    import cv2 as _cv2
    img = _cv2.imread(img_path)
    if img is None:
        raise HTTPException(status_code=400, detail="cannot read image")
    h, w = img.shape[:2]

    try:
        import pipeline as _pipeline
        results = _pipeline.pose_model(img, conf=0.25, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"model error: {e}")

    found = False
    keypoints = [{"x": 0.5, "y": 0.5, "v": 0} for _ in range(17)]
    if results and len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints) > 0:
        # 가장 큰 사람 선택
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            areas = boxes.xywh.cpu().numpy()[:, 2:4].prod(axis=1)
            idx = int(areas.argmax())
        else:
            idx = 0
        kpts = results[0].keypoints.data[idx].cpu().numpy()  # (17, 3)
        keypoints = []
        for i in range(17):
            x, y, c = float(kpts[i, 0]), float(kpts[i, 1]), float(kpts[i, 2])
            v = 2 if c >= 0.5 else (1 if c >= 0.2 else 0)
            keypoints.append({
                "x": max(0.0, min(1.0, x / max(w, 1))),
                "y": max(0.0, min(1.0, y / max(h, 1))),
                "v": v,
            })
        found = any(k["v"] > 0 for k in keypoints)

    return JSONResponse({
        "found": found,
        "image": {"width": w, "height": h},
        "keypoints": keypoints,
    })


class PoseKeypoint(BaseModel):
    x: float  # 0..1 (정규화)
    y: float  # 0..1
    v: int    # 0=없음, 1=가려짐(추정), 2=보임


class SavePoseLabelsBody(BaseModel):
    keypoints: list[PoseKeypoint]


@app.post("/pose/labels/{session_id}/{filename}")
def save_pose_labels(session_id: str, filename: str, body: SavePoseLabelsBody):
    """YOLO 포즈 형식으로 라벨 저장.
    파일 라인: class cx cy w h px1 py1 v1 ... px17 py17 v17
    bbox 는 visible(v>=1) 키포인트들의 min/max 에 10% 패딩으로 자동 산출.
    """
    base, images_dir, labels_dir = _ensure_pose_session(session_id)
    img_path = os.path.join(images_dir, filename)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="image not found")
    if len(body.keypoints) != 17:
        raise HTTPException(status_code=400, detail="keypoints must be 17")

    visible = [(k.x, k.y) for k in body.keypoints if k.v > 0]
    if len(visible) < 4:
        raise HTTPException(status_code=400, detail="최소 4개 이상의 키포인트가 라벨링되어야 합니다.")

    xs = [p[0] for p in visible]
    ys = [p[1] for p in visible]
    x_min, x_max = max(0.0, min(xs)), min(1.0, max(xs))
    y_min, y_max = max(0.0, min(ys)), min(1.0, max(ys))
    pad_x = (x_max - x_min) * 0.1 + 0.02
    pad_y = (y_max - y_min) * 0.1 + 0.02
    x_min = max(0.0, x_min - pad_x)
    y_min = max(0.0, y_min - pad_y)
    x_max = min(1.0, x_max + pad_x)
    y_max = min(1.0, y_max + pad_y)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    bw = x_max - x_min
    bh = y_max - y_min

    parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for k in body.keypoints:
        x = max(0.0, min(1.0, k.x))
        y = max(0.0, min(1.0, k.y))
        v = 0 if k.v not in (0, 1, 2) else int(k.v)
        # YOLO 포즈는 v=0 이면 (0,0,0) 으로 적는 것이 관례 (학습에서 무시됨)
        if v == 0:
            parts += ["0.000000", "0.000000", "0"]
        else:
            parts += [f"{x:.6f}", f"{y:.6f}", str(v)]

    line = " ".join(parts)
    lbl_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
    with open(lbl_path, "w") as f:
        f.write(line + "\n")
    return JSONResponse({"status": "saved", "line": line})


@app.get("/pose/labels/{session_id}/{filename}")
def get_pose_labels(session_id: str, filename: str):
    base, images_dir, labels_dir = _ensure_pose_session(session_id)
    lbl_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
    if not os.path.exists(lbl_path):
        return JSONResponse({"keypoints": None})
    with open(lbl_path, "r") as f:
        line = next((ln.strip() for ln in f if ln.strip() and not ln.startswith("#")), "")
    if not line:
        return JSONResponse({"keypoints": None})
    parts = line.split()
    if len(parts) < 5 + 17 * 3:
        return JSONResponse({"keypoints": None})
    keypoints = []
    for i in range(17):
        ofs = 5 + i * 3
        try:
            x = float(parts[ofs])
            y = float(parts[ofs + 1])
            v = int(float(parts[ofs + 2]))
        except (ValueError, IndexError):
            x, y, v = 0.0, 0.0, 0
        keypoints.append({"x": x, "y": y, "v": v})
    return JSONResponse({"keypoints": keypoints})


def _run_pose_training(
    train_job_id: str,
    session_id: str,
    epochs: int,
    imgsz: int,
    initial_weights: str = "",
    auto_activate: bool = False,
):
    """포즈 모델 파인튜닝 (yolo11l-pose.pt 또는 사용자 지정 가중치 사용).
    학습 데이터: 현재 세션의 라벨링된 이미지만 사용 (val 자동 80:20 분할).
    """
    from ultralytics import YOLO
    import yaml
    from pathlib import Path as _Path

    train_jobs[train_job_id]["status"] = "processing"
    base = _pose_session_dir(session_id)
    src_images = os.path.join(base, "images")
    src_labels = os.path.join(base, "labels")
    log = train_jobs[train_job_id]["log"]

    work = os.path.join(base, "_work")
    if os.path.exists(work):
        shutil.rmtree(work)
    train_img = os.path.join(work, "train", "images")
    train_lbl = os.path.join(work, "train", "labels")
    val_img = os.path.join(work, "val", "images")
    val_lbl = os.path.join(work, "val", "labels")
    for d in [train_img, train_lbl, val_img, val_lbl]:
        os.makedirs(d, exist_ok=True)

    pairs = []
    for fn in sorted(os.listdir(src_images)):
        if not fn.lower().endswith(IMG_EXTS):
            continue
        stem = os.path.splitext(fn)[0]
        lbl = os.path.join(src_labels, stem + ".txt")
        if os.path.exists(lbl) and _count_pose_label(lbl):
            pairs.append((os.path.join(src_images, fn), lbl))
    log.append(f"라벨링된 포즈 이미지 {len(pairs)}장 수집")

    if len(pairs) < 2:
        train_jobs[train_job_id]["status"] = "failed"
        train_jobs[train_job_id]["error"] = "최소 2장 이상의 라벨링된 이미지가 필요합니다 (train/val 분할용)."
        return

    # 80:20 분할
    split = max(1, int(len(pairs) * 0.8))
    tr_pairs = pairs[:split]
    va_pairs = pairs[split:] if len(pairs) - split >= 1 else pairs[-1:]

    def copy_pair(pair, dst_img_dir, dst_lbl_dir):
        img_src, lbl_src = pair
        fn = os.path.basename(img_src)
        shutil.copy2(img_src, os.path.join(dst_img_dir, fn))
        new_lbl = os.path.splitext(fn)[0] + ".txt"
        shutil.copy2(lbl_src, os.path.join(dst_lbl_dir, new_lbl))

    for p in tr_pairs:
        copy_pair(p, train_img, train_lbl)
    for p in va_pairs:
        copy_pair(p, val_img, val_lbl)

    # data.yaml (포즈 형식: kpt_shape, flip_idx 필수)
    work_abs = os.path.abspath(work).replace("\\", "/")
    train_img_abs = os.path.abspath(train_img).replace("\\", "/")
    val_img_abs = os.path.abspath(val_img).replace("\\", "/")
    data_yaml = {
        "train": train_img_abs,
        "val": val_img_abs,
        "nc": 1,
        "names": ["person"],
        "kpt_shape": [17, 3],
        "flip_idx": COCO_KPT_FLIP_IDX,
    }
    yaml_path = os.path.abspath(os.path.join(work, "data.yaml")).replace("\\", "/")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, allow_unicode=True)
    log.append(f"data.yaml: {yaml_path}")
    log.append(f"  train: {train_img_abs} ({len(tr_pairs)}장)")
    log.append(f"  val:   {val_img_abs} ({len(va_pairs)}장)")

    # Ultralytics DATASETS_DIR 패치 (한글/datasets_dir 재해석 우회)
    try:
        import ultralytics.utils as _ul_utils
        import ultralytics.data.utils as _ul_data_utils
        from ultralytics.utils import SETTINGS as _ULSETTINGS
        _ULSETTINGS.update({"datasets_dir": work_abs})
        _ul_utils.DATASETS_DIR = _Path(work_abs)
        _ul_data_utils.DATASETS_DIR = _Path(work_abs)
    except Exception as _e:
        log.append(f"warning: DATASETS_DIR 패치 실패 ({_e})")

    try:
        weights_to_load = "yolo11l-pose.pt"
        if initial_weights:
            iw = initial_weights.replace("\\", "/")
            if os.path.exists(iw):
                weights_to_load = iw
                log.append(f"초기 가중치 로드 (fine-tuning): {weights_to_load}")
            else:
                log.append(f"경��: 초기 가중치 파일을 찾지 못했습니다 ({iw}). yolo11l-pose.pt 사용")
        else:
            log.append("초기 가중치: yolo11l-pose.pt (사전학습 포즈 모델)")
        model = YOLO(weights_to_load)

        total_epochs = epochs

        def on_train_epoch_end(trainer):
            ep = trainer.epoch + 1
            pct = int(ep / total_epochs * 100)
            train_jobs[train_job_id]["progress"] = pct
            line = f"Epoch {ep}/{total_epochs} | progress: {pct}%"
            metrics = trainer.metrics if hasattr(trainer, "metrics") else {}
            if metrics:
                for k, v in metrics.items():
                    line += f" | {k}: {v:.4f}" if isinstance(v, float) else f" | {k}: {v}"
            train_jobs[train_job_id]["log"].append(line)

        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        result = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            workers=0,
            project=os.path.abspath(os.path.join(work, "runs")).replace("\\", "/"),
            name="pose_train",
            exist_ok=True,
        )
        best_path = str(result.save_dir / "weights" / "best.pt").replace("\\", "/") if result else ""
        train_jobs[train_job_id]["status"] = "done"
        train_jobs[train_job_id]["progress"] = 100
        train_jobs[train_job_id]["best_model"] = best_path
        train_jobs[train_job_id]["log"].append(f"학습 완료. 모델 저장: {best_path}")

        # 자동 활성화: params.json 의 pose_model_path 갱신 + pipeline 핫스왑
        if auto_activate:
            if best_path and os.path.exists(best_path):
                try:
                    cur = load_params()
                    cur["pose_model_path"] = best_path
                    save_params(cur)
                    _apply_params_to_pipeline(cur)
                    train_jobs[train_job_id]["log"].append(
                        "활성 포즈 모델로 자동 적용됨 (params.json 갱신, 재시작 후에도 유지)"
                    )
                    train_jobs[train_job_id]["activated"] = True
                except Exception as ee:
                    train_jobs[train_job_id]["log"].append(f"자동 활성화 실패: {ee}")
            else:
                train_jobs[train_job_id]["log"].append(
                    f"자동 활성화 건너뜀 - best.pt 파일 확인 실패 ({best_path})"
                )
    except Exception as e:
        train_jobs[train_job_id]["status"] = "failed"
        train_jobs[train_job_id]["error"] = str(e)
        train_jobs[train_job_id]["log"].append(f"오류: {e}")


@app.post("/pose/train/{session_id}")
def start_pose_training(
    session_id: str,
    epochs: int = 50,
    imgsz: int = 640,
    initial_weights: str = "",
    auto_activate: bool = False,
):
    """포즈 키포인트 라벨로 yolo11l-pose 파인튜닝 시작.
    auto_activate=True 면 학습 완료 시 best.pt를 활성 포즈 모델로 자동 설정.
    """
    base = _pose_session_dir(session_id)
    if not os.path.exists(base):
        raise HTTPException(status_code=404, detail="pose session not found")

    effective = _migrate_to_ascii_pose_session(session_id)
    train_job_id = str(uuid.uuid4())
    train_jobs[train_job_id] = {
        "status": "queued",
        "error": "",
        "progress": 0,
        "log": (
            [f"세션 ID 비-ASCII 포함 → ASCII 사본 '{effective}' 으로 학습"]
            if effective != session_id else []
        ),
        "session_id": effective,
        "best_model": None,
        "activated": False,
    }
    thread = threading.Thread(
        target=_run_pose_training,
        args=(train_job_id, effective, epochs, imgsz, initial_weights, auto_activate),
        daemon=True,
    )
    thread.start()
    return JSONResponse({"train_job_id": train_job_id, "effective_session_id": effective})


def _migrate_to_ascii_pose_session(session_id: str) -> str:
    if _ascii_safe(session_id):
        return session_id
    new_sid = "".join(c for c in session_id if c.isascii() and (c.isalnum() or c in "-_"))
    if not new_sid:
        new_sid = datetime.now().strftime("pose_%Y%m%d_%H%M%S")
    if os.path.exists(_pose_session_dir(new_sid)):
        new_sid = f"{new_sid}_{uuid.uuid4().hex[:4]}"
    src = _pose_session_dir(session_id)
    dst = _pose_session_dir(new_sid)
    shutil.copytree(src, dst)
    return new_sid


# ────────────────────────────────────────────────────���───────────────────────
# ngrok 자동 터널 (python app.py --ngrok 또는 --ngrok-token TOKEN 으로 실행)
# ────────────────────────────────────────────────────────────────────────────
def _start_ngrok(port: int, token: str = ""):
    """ngrok 터널을 백그라운드에서 시작하고 공개 URL을 _ngrok_url에 저장"""
    global _ngrok_url
    try:
        import ngrok as ngrok_sdk
        if token:
            ngrok_sdk.set_auth_token(token)
        listener = ngrok_sdk.forward(port, authtoken_from_env=not bool(token))
        _ngrok_url = listener.url()
        print(f"\n{'='*60}")
        print(f"  ngrok 터널 활성화")
        print(f"  공개 URL: {_ngrok_url}")
        print(f"")
        print(f"  웹 브라우저에서 아래 URL을 로컬 서버 주소로 입력하세요:")
        print(f"  https://climbingposeanalyzer-local.vercel.app")
        print(f"  -> 상단 '로컬 서버' 입력란에: {_ngrok_url}")
        print(f"{'='*60}\n")
    except ImportError:
        print("[ERROR] ngrok 패키지가 없습니다. 설치: pip install ngrok")
        print("        또는 https://ngrok.com 에서 직접 다운로드 후 실행하세요.")
    except Exception as e:
        print(f"[ERROR] ngrok 시작 실패: {e}")
        print("        NGROK_AUTHTOKEN 환경변수를 설정하거나 --ngrok-token 옵션을 사용하세요.")


# ────────────────────────────────────────────────────────────────────────────
# CLI 진입점
# 실행 예시:
#   python app.py                          # localhost:8000 (로컬 전용)
#   python app.py --ngrok                  # ngrok 터널 자동 시작 (NGROK_AUTHTOKEN 환경변수 필요)
#   python app.py --ngrok --token YOUR_TOKEN
#   python app.py --port 8080 --ngrok
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Climbing Pose Analyzer 로컬 서버")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트 (기본: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="바인딩 호스트 (기본: 0.0.0.0)")
    parser.add_argument("--ngrok", action="store_true", help="ngrok 터널 자동 시작")
    parser.add_argument("--token", type=str, default="", help="ngrok authtoken (NGROK_AUTHTOKEN 환경변수 대안)")
    args = parser.parse_args()

    if args.ngrok:
        ngrok_token = args.token or os.environ.get("NGROK_AUTHTOKEN", "")
        t = threading.Thread(target=_start_ngrok, args=(args.port, ngrok_token), daemon=True)
        t.start()
        # URL 출력 대기
        import time
        time.sleep(2)

    print(f"\n로컬 서버 시작: http://localhost:{args.port}")
    print(f"Vercel 웹: https://climbingposeanalyzer-local.vercel.app\n")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
