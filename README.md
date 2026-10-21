# CPAP (ClimbingPoseAnalyzerPro)

클라이밍 영상에서 **무게중심(COG) 궤적과 이동 속도(m/s)** 를 분석하는 강습 보조 도구.
Vercel에 배포된 Next.js 웹에서 사용자 PC의 분석 서버(FastAPI + YOLO)에 연결해 사용합니다.

---

## 아키텍처

```
[Next.js 웹 - Vercel 배포]  (이 저장소 루트)
        ↕  REST API
[분석 서버 - 사용자 PC]      (server/)
  1. extract_raw.py   포즈 탐지 원본값 추출 (GPU)
  2. refine.py        클라이머 추적 · 좌우 교정 · 오류 관절 보간 · 접촉 구간 고정 · 분절 질량 COG
  3. analyze.py       키·윙스팬으로 px→m 스케일 (보정 자세 자동 감지) · 원근 보정 · COG 속도
  4. render_lesson.py 강습용 영상 (연한 COG 궤적 + 속도 그래프)
```

---

## 분석 서버 실행 (Windows, C:\Project 기준)

### 1. 의존성 (최초 1회)

```bash
cd C:\Project
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # GPU용
pip install -r server\requirements.txt
```

### 2. 포즈 모델 지정 (선택)

`server\config.example.json` 을 `server\config.json` 으로 복사해 `pose_model_path` 를 수정합니다.
없으면 `C:\Project\params.json` 의 `pose_model_path`, 그것도 없으면 `yolo11l-pose.pt` 를 씁니다.

### 3. 실행

```bash
.venv\Scripts\python server\app.py            # 같은 PC에서: http://localhost:8000
.venv\Scripts\python server\app.py --ngrok    # Vercel 웹에서 접속 (NGROK_AUTHTOKEN 필요)
```

> HTTPS인 Vercel 페이지에서는 브라우저 보안 정책 때문에 `http://localhost` 에 바로 연결할 수 없습니다.
> `--ngrok` 으로 실행하고, 터미널에 나온 `https://….ngrok-free.app` 주소를 웹 상단 '로컬 서버' 입력란에 넣으세요.
> 무료 플랜은 실행할 때마다 주소가 바뀝니다.

ngrok 토큰 설정: https://ngrok.com 가입 → Authtoken 복사 →
`set NGROK_AUTHTOKEN=토큰` (또는 `--token 토큰`). **토큰은 저장소에 커밋하지 마세요.**

분석 결과와 작업 기록은 `server\data\` 에 저장되어 서버를 재시작해도 남습니다.

---

## 촬영 가이드

1. 등반 전, 벽에 등을 붙이고 카메라를 보며 **양팔을 어깨 높이로 쭉 펴고 2초** (보정 자세)
2. 삼각대 고정, 촬영 중 위치·줌 변경 금지
3. 머리부터 발끝까지 등반 전 구간이 화면에 들어오게
4. 가능하면 1080p (30fps면 충분)

---

## 기능

| 페이지 | 기능 |
|---|---|
| 영상 분석 | 클라이머 프로필(이름·키·윙스팬) → 영상 업로드 → 진행률 → COG 영상 · 속도 그래프 · 동작별 최고 속도 |
| 영상 비교 | 분석된 영상 A/B 동시 재생, 줌, 속도 조절 |

---

## API

| Method | Path | 설명 |
|---|---|---|
| GET | `/ping` | 서버 상태 |
| POST | `/upload` | 영상 업로드 (`file`, `height_cm`, `wingspan_cm`, `climber_name`) |
| GET | `/status/{job_id}` | 상태 · 진행률 · 단계 · 대기 순번 |
| GET | `/summary/{job_id}` | 분석 요약 (등반 구간, 평균/최고 속도, 동작 목록, 속도 시계열) |
| GET | `/video/{job_id}` | 결과 영상 |
| GET | `/jobs` | 최근 작업 목록 |
| GET | `/videos` | 완료된 결과 영상 목록 (비교 화면용) |

---

## 이전 버전

루트의 `app.py`, `pipeline.py`, `cog.py`, `datalearning.py` 는 이전 버전(홀드·포즈 학습 포함) 서버입니다.
새 서버(`server/`)는 학습한 모델 파일만 읽어서 쓰므로, 학습 기능 없이도 동작합니다.
