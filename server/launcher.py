"""
CPAP 서버 원클릭 실행기 ("CPAP 서버 시작.bat" 이 이 파일을 실행한다)

1) 필요한 패키지 확인 → 없으면 설치 (NVIDIA GPU가 있으면 CUDA용 PyTorch)
2) ngrok 토큰 확인 → 없으면 처음 한 번만 물어보고 server/config.json 에 저장
3) 이미 서버가 켜져 있으면 연결 페이지만 다시 연다
4) 서버 + ngrok 실행 → 외부 주소가 나오면 연결 페이지(휴대폰 QR 코드) 자동으로 열기
5) 처음 실행 때 바탕화면 바로가기 만들기 (선택)

이 창을 닫거나 Ctrl+C 를 누르면 서버가 꺼진다.
"""
import os
import sys
import json
import time
import shutil
import subprocess
import webbrowser
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG = os.path.join(HERE, "config.json")
PORT = 8000
LOCAL = f"http://localhost:{PORT}"
TOKEN_PAGE = "https://dashboard.ngrok.com/get-started/your-authtoken"
DOMAIN_PAGE = "https://dashboard.ngrok.com/domains"

REQUIRED = {  # import 이름: pip 패키지 이름
    "fastapi": "fastapi", "uvicorn": "uvicorn", "multipart": "python-multipart",
    "numpy": "numpy", "scipy": "scipy", "cv2": "opencv-python",
    "torch": "torch", "ultralytics": "ultralytics", "ngrok": "ngrok",
}


def say(msg=""):
    print(msg, flush=True)


def step(n, msg):
    say(f"\n[{n}/4] {msg}")


# ────────────────────────────────────────────────────────────────────────────
def load_cfg():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cfg(cfg):
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def ask(prompt, default=""):
    try:
        v = input(prompt).strip()
    except EOFError:
        return default
    return v or default


# ────────────────────────────────────────────────────────────────────────────
def ensure_packages():
    missing = []
    for mod, pkg in REQUIRED.items():
        try:
            __import__(mod)
        except Exception:
            missing.append(pkg)
    if not missing:
        say("  필요한 패키지가 모두 설치되어 있습니다.")
        return
    say(f"  설치가 필요한 패키지: {', '.join(missing)}")
    pip = [sys.executable, "-m", "pip", "install"]
    if "torch" in missing:
        if shutil.which("nvidia-smi"):
            say("  NVIDIA GPU 감지 → CUDA용 PyTorch 설치 (2~3GB, 몇 분 걸립니다)")
            subprocess.check_call(pip + ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118"])
        else:
            say("  NVIDIA GPU 없음 → CPU용 PyTorch 설치")
            subprocess.check_call(pip + ["torch", "torchvision"])
    rest = [p for p in missing if p != "torch"]
    if rest:
        subprocess.check_call(pip + rest)
    say("  설치 완료")


def gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            say(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            say("  GPU를 찾지 못했습니다. CPU로 분석하므로 느릴 수 있습니다.")
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────────────────
def find_token(cfg):
    if os.environ.get("NGROK_AUTHTOKEN"):
        return os.environ["NGROK_AUTHTOKEN"]
    if cfg.get("ngrok_token"):
        return cfg["ngrok_token"]
    home = os.path.expanduser("~")
    for path in (os.path.join(os.environ.get("LOCALAPPDATA", os.path.join(home, "AppData", "Local")), "ngrok", "ngrok.yml"),
                 os.path.join(home, "AppData", "Roaming", "ngrok", "ngrok.yml"),
                 os.path.join(home, ".config", "ngrok", "ngrok.yml")):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("authtoken:"):
                        tok = line.split(":", 1)[1].strip().strip("'\"")
                        if tok:
                            return tok
        except OSError:
            continue
    return ""


def ensure_ngrok(cfg):
    tok = find_token(cfg)
    if tok:
        say("  ngrok 토큰이 설정되어 있습니다.")
    else:
        say("  휴대폰에서 접속하려면 ngrok 토큰이 필요합니다 (처음 한 번만).")
        say(f"  브라우저에서 ngrok 토큰 페이지를 엽니다: {TOKEN_PAGE}")
        say("  (계정이 없으면 무료 가입 후, 'Your Authtoken' 을 복사하세요)")
        webbrowser.open(TOKEN_PAGE)
        tok = ask("\n  토큰을 붙여넣고 Enter (건너뛰려면 그냥 Enter): ")
        if tok:
            cfg["ngrok_token"] = tok
            save_cfg(cfg)
            say("  토큰을 server\\config.json 에 저장했습니다. 다음부터는 묻지 않습니다.")
        else:
            say("  건너뜀: 이번에는 이 PC에서만 접속할 수 있습니다.")
            return False

    if "ngrok_domain" not in cfg:
        say("\n  [선택] ngrok 무료 계정에는 고정 주소(도메인)가 1개 있습니다.")
        say("  고정 주소를 쓰면 휴대폰 즐겨찾기가 계속 유지되어 매번 QR을 찍지 않아도 됩니다.")
        say(f"  확인: {DOMAIN_PAGE}  (예: something-word-1234.ngrok-free.app)")
        dom = ask("  고정 주소를 붙여넣고 Enter (없으면 그냥 Enter): ")
        cfg["ngrok_domain"] = dom.replace("https://", "").rstrip("/")
        save_cfg(cfg)
    return True


# ────────────────────────────────────────────────────────────────────────────
def get_json(path, timeout=2):
    req = urllib.request.Request(LOCAL + path, headers={"ngrok-skip-browser-warning": "true"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def running_server():
    """8000 포트에 이미 무엇이 떠 있는지: 'v3' / 'other' / None"""
    try:
        info = get_json("/ping")
        return "v3" if str(info.get("version", "")).startswith("3") else "other"
    except urllib.error.HTTPError:
        return "other"
    except Exception:
        return None


def make_shortcut(cfg):
    if os.name != "nt" or cfg.get("shortcut_asked"):
        return
    cfg["shortcut_asked"] = True
    save_cfg(cfg)
    if ask("\n  바탕화면에 'CPAP 서버 시작' 바로가기를 만들까요? (Y/n): ", "y").lower() not in ("y", "yes", "ㅛ"):
        return
    bat = os.path.join(ROOT, "CPAP 서버 시작.bat")
    ps = (
        "$d=[Environment]::GetFolderPath('Desktop');"
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'CPAP 서버 시작.lnk'));"
        f"$s.TargetPath='{bat}';$s.WorkingDirectory='{ROOT}';"
        "$s.IconLocation='imageres.dll,-1023';$s.Save()"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)
        say("  바탕화면에 바로가기를 만들었습니다.")
    except Exception as e:
        say(f"  바로가기를 만들지 못했습니다: {e}")


# ────────────────────────────────────────────────────────────────────────────
def main():
    say("=" * 60)
    say("  CPAP 분석 서버 시작")
    say("=" * 60)

    state = running_server()
    if state == "v3":
        say("\n이미 서버가 실행 중입니다. 연결 페이지를 엽니다.")
        webbrowser.open(LOCAL + "/connect")
        return
    if state == "other":
        say(f"\n[!] {PORT} 포트를 다른 프로그램(예전 서버 app.py 등)이 쓰고 있습니다.")
        say("    그 창을 닫은 뒤 다시 실행해 주세요.")
        return

    step(1, "실행 환경 확인")
    say(f"  Python: {sys.executable}")
    ensure_packages()
    gpu_info()

    cfg = load_cfg()
    step(2, "외부 접속(ngrok) 설정")
    use_ngrok = ensure_ngrok(cfg)

    step(3, "서버 시작")
    cmd = [sys.executable, os.path.join(HERE, "app.py"), "--port", str(PORT)]
    if use_ngrok:
        cmd.append("--ngrok")
    proc = subprocess.Popen(cmd, cwd=ROOT)

    # 서버가 뜨고 ngrok 주소가 나올 때까지 기다림
    ngrok_url, err, t0 = "", "", time.time()
    while time.time() - t0 < 90:
        if proc.poll() is not None:
            say("\n[!] 서버가 시작 직후 종료되었습니다. 위의 오류 메시지를 확인하세요.")
            return
        try:
            tun = get_json("/tunnel")
            ngrok_url, err = tun.get("ngrok_url", ""), tun.get("error", "")
            if ngrok_url or err or not use_ngrok:
                break
        except Exception:
            pass
        time.sleep(1)

    step(4, "연결 페이지 열기")
    webbrowser.open(LOCAL + "/connect")
    say("")
    say("=" * 60)
    say(f"  이 PC:   {LOCAL}")
    if ngrok_url:
        say(f"  휴대폰:  {ngrok_url}")
        say("  → 열린 연결 페이지의 QR 코드를 휴대폰 카메라로 찍으세요.")
        try:
            subprocess.run("clip", input=ngrok_url.encode("ascii"), check=False, shell=True)
            say("  (휴대폰 주소를 클립보드에 복사했습니다)")
        except Exception:
            pass
    elif use_ngrok:
        say(f"  휴대폰:  연결 실패 - {err or '시간 초과'}")
        say("           토큰이 잘못됐다면 server\\config.json 의 ngrok_token 을 지우고 다시 실행하세요.")
    say("")
    say("  이 창을 닫거나 Ctrl+C 를 누르면 서버가 꺼집니다.")
    say("=" * 60)

    make_shortcut(cfg)

    try:
        proc.wait()
    except KeyboardInterrupt:
        say("\n서버를 끕니다...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        say(f"\n[!] 패키지 설치에 실패했습니다: {e}")
        say("    인터넷 연결을 확인하고 다시 실행해 주세요.")
    except KeyboardInterrupt:
        pass
