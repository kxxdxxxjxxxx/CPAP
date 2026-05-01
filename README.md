# CPAP

# ClimbingPoseAnalyzer

클라이밍 영상 포즈 분석 플랫폼. Vercel에 배포된 Next.js 웹에서 로컬 FastAPI 서버를 연결하여 사용합니다.

---

## 아키텍처

```
[Next.js 웹 - Vercel 배포]
        ↕  REST API (CORS 허용)
[로컬 FastAPI 서버 - 사용자 PC]
  ├── YOLO 포즈 분석 (pipeline.py)
  ├── 홀드 탐지 학습 (YOLOv11 파인튜닝)
  └── 파라미터 저장 (params.json)
```

---

## 로컬 서버 실행 방법

### 1. 의존성 설치

```bash
pip install fastapi uvicorn ultralytics opencv-python numpy pydantic pyyaml python-multipart ngrok
```

### 2-A. 로컬 전용 실행 (같은 PC에서 접속 시)

```bash
python app.py
# 또는
uvicorn app:app --host 0.0.0.0 --port 8000
```

웹 연결 바 입력: `http://localhost:8000`

---

### 2-B. Vercel 배포 웹에서 접속 (ngrok 필수)

> **이유:** `https://climbingposeanalyzer-local.vercel.app` 같은 HTTPS 페이지에서는
> 브라우저 보안 정책(Mixed Content)으로 인해 `http://localhost` 직접 연결이 차단됩니다.
> ngrok을 사용하면 로컬 서버에 HTTPS 공개 URL을 발급받을 수 있습니다.

#### ngrok 계정 설정 (최초 1회)

1. https://ngrok.com/signup 에서 무료 계정 생성
2. 대시보드 → Your Authtoken 복사
3. 아래 중 하나로 토큰 설정:

```bash
# 방법 A: 환경변수
export NGROK_AUTHTOKEN=your_token_here   # macOS/Linux
set NGROK_AUTHTOKEN=your_token_here      # Windows

# 방법 B: 실행 시 직접 전달
python app.py --ngrok --token your_token_here
```

#### ngrok 포함 서버 실행

```bash
python app.py --ngrok
```

터미널 출력 예시:
```
============================================================
  ngrok 터널 활성화
  공개 URL: https://a1b2c3d4.ngrok-free.app

  웹 브라우저에서 아래 URL을 로컬 서버 주소로 입력하세요:
  https://climbingposeanalyzer-local.vercel.app
  -> 상단 '로컬 서버' 입력란에: https://a1b2c3d4.ngrok-free.app
============================================================
```

웹 연결 바 입력: `https://a1b2c3d4.ngrok-free.app`

> 무료 플랜은 세션마다 URL이 변경됩니다. 서버 재시작 후 웹에서 URL을 다시 입력하세요.
> 고정 도메인이 필요하면 ngrok 유료 플랜 또는 [고정 도메인](https://ngrok.com/docs/ngrok-agent/config/#tunnels) 옵션을 사용하세요.

---

## 기능

| 페이지 | 기능 |
|---|---|
| 영상 분석 | MP4 업로드 → 포즈+홀드 분석 → 결과 다운로드 |
| 영상 비교 | 분석된 영상 A/B 동시 재생, 줌, 속도 조절 |
| 홀드 학습 | 이미지+YOLO 라벨 업로드 → YOLOv11 파인튜닝 |
| 파라미터 튜닝 | 폐색 임계값, 탐색 반경, 스무딩, 신뢰도 등 실시간 조정 |

---

## API 엔드포인트

| Method | Path | 설명 |
|---|---|---|
| GET | `/ping` | 서버 상태 확인 |
| GET | `/params` | 현재 파라미터 조회 |
| POST | `/params` | 파라미터 저장 및 즉시 적용 |
| POST | `/upload` | 영상 업로드 및 분석 시작 |
| GET | `/status/{job_id}` | 분석 작업 상태 조회 |
| GET | `/video/{job_id}` | 분석 결과 영상 다운로드 |
| GET | `/videos` | 분석된 영상 목록 |
| POST | `/train/upload` | 학습 데이터 업로드 |
| POST | `/train/start/{session_id}` | 학습 시작 |
| GET | `/train/status/{train_job_id}` | 학습 진행률 조회 |
| GET | `/train/download/{train_job_id}` | 학습된 best.pt 다운로드 |
