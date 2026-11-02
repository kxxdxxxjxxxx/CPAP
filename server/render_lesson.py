"""
강습용 영상: 무게중심만 표시 (연한 궤적) + COG 속도 + 하단 속도 그래프

단독 실행:  python render_lesson.py 영상.raw.npz 영상.mp4 --height 175 --wingspan 178 --out lesson.mp4
서버에서는 render(A, video, out) 와 summarize(A) 를 직접 호출한다.
"""
import os
import sys
import shutil
import argparse
import subprocess
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze as AN  # noqa: E402

# ── 보기 설정 ──
TRAIL_SEC = 3.0          # 최근 궤적 길이 (초)
TRAIL_ALPHA = 0.35       # 최근 궤적 진하기 (0~1)
GHOST_ALPHA = 0.12       # 전체 경로(지나온 길) 진하기
SPEED_FAST = 1.2         # 이 속도(m/s) 이상이면 궤적 색이 가장 진한 주황
GRAPH_H_RATIO = 0.125    # 하단 그래프 높이 = 영상 높이의 12.5%
REF_SIZE = 720           # 글자·선 굵기 기준 (영상 짧은 변이 720px일 때 1배)

def speed_color(v):
    """느림=흰색 → 빠름=주황 (BGR)"""
    t = float(np.clip(np.nan_to_num(v) / SPEED_FAST, 0, 1))
    white, orange = np.array([255, 255, 255]), np.array([40, 150, 255])
    return tuple(int(c) for c in (1 - t) * white + t * orange)


def ffmpeg_exe():
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _direction(vx, vy, v):
    if not np.isfinite(v) or v <= 0.15:
        return ""
    h = ("RIGHT" if vx > 0 else "LEFT") if abs(vx) > 0.5 * abs(vy) else ""
    u = ("UP" if vy > 0 else "DOWN") if abs(vy) > 0.5 * abs(vx) else ""
    return "-".join(x for x in (u, h) if x)


def summarize(A):
    """화면에 보여줄 분석 요약 (JSON 직렬화 가능)."""
    fps = A["fps"]
    s, e = A["seg"]
    sp = A["speed"]
    si = A["scale_info"]
    ko = {"UP": "위", "DOWN": "아래", "LEFT": "왼쪽", "RIGHT": "오른쪽"}
    moves = []
    for p, v in A["moves"]:
        d = _direction(A["vel"][p, 0], A["vel"][p, 1], v)
        moves.append({"time": round(p / fps, 2), "speed": round(v, 2),
                      "direction": "-".join(ko[x] for x in d.split("-") if x)})
    out = {
        "climb_start": round(s / fps, 2), "climb_end": round(e / fps, 2),
        "duration": round((e - s) / fps, 2),
        "mean_speed": round(float(np.nanmean(sp[s:e])), 3),
        "max_speed": round(float(np.nanmax(sp[s:e])), 3),
        "px_per_m": round(A["px_per_m"], 1),
        "scale_method": si.get("method", "body_proportion"),
        "calibration_detected": si.get("method") == "calibration_pose",
        "moves": moves,
        "speed_series": [None if not np.isfinite(x) else round(float(x), 3) for x in sp[s:e:max(1, int(fps // 6))]],
        "series_dt": round(max(1, int(fps // 6)) / fps, 4),
    }
    if out["calibration_detected"]:
        c0, c1 = si["calib_frames"]
        out["calibration_time"] = [round(c0 / fps, 2), round(c1 / fps, 2)]
    return out


def render(A, video, out_path, progress=None):
    """분석 결과 A 로 강습용 영상을 만든다. progress(done, total) 콜백."""
    r, fps = A["refined"], A["fps"]
    s, e = A["seg"]
    cog, speed, vel = r["cog"], A["speed"], A["vel"]
    moves = {p for p, _ in A["moves"]}

    cap = cv2.VideoCapture(video)
    W, H = int(cap.get(3)), int(cap.get(4))
    tmp = os.path.splitext(out_path)[0] + "_raw.mp4"
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    TR = int(TRAIL_SEC * fps)

    # 그래프 좌표계 (등반 구간 전체를 가로 폭에 맞춤)
    # 해상도가 달라도 화면에서 차지하는 비율이 같도록 모든 크기를 영상 크기에 비례시킨다
    u = min(W, H) / REF_SIZE
    px = lambda v: max(1, int(round(v * u)))
    fs = lambda v: v * u                      # 글자 크기
    gx0, gx1 = px(50), W - px(20)
    gy1 = H - px(20)
    gy0 = gy1 - int(H * GRAPH_H_RATIO)
    vmax = max(1.5, float(np.nanmax(speed[s:e])) * 1.05)
    tx = lambda t: int(gx0 + (t - s) / max(1, e - 1 - s) * (gx1 - gx0))
    vy = lambda v: int(gy1 - np.clip(np.nan_to_num(v) / vmax, 0, 1) * (gy1 - gy0))

    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if s <= t < e:
            # 2) 지나온 전체 경로 (아주 연하게)
            ov = frame.copy()
            pts = cog[s:t + 1]
            for i in range(1, len(pts)):
                if np.isfinite(pts[i - 1:i + 1]).all():
                    cv2.line(ov, tuple(pts[i - 1].astype(int)), tuple(pts[i].astype(int)), (255, 255, 255), px(2), cv2.LINE_AA)
            cv2.addWeighted(ov, GHOST_ALPHA, frame, 1 - GHOST_ALPHA, 0, frame)

            # 3) 최근 궤적 (속도에 따라 색, 연하게)
            ov = frame.copy()
            t0 = max(s, t - TR)
            for i in range(t0 + 1, t + 1):
                if np.isfinite(cog[i - 1:i + 1]).all():
                    cv2.line(ov, tuple(cog[i - 1].astype(int)), tuple(cog[i].astype(int)),
                             speed_color(speed[i]), px(4), cv2.LINE_AA)
            cv2.addWeighted(ov, TRAIL_ALPHA, frame, 1 - TRAIL_ALPHA, 0, frame)

            # 4) 현재 COG (작은 점)
            if np.isfinite(cog[t]).all():
                c = tuple(cog[t].astype(int))
                cv2.circle(frame, c, px(7), (255, 255, 255), px(2), cv2.LINE_AA)
                cv2.circle(frame, c, px(4), speed_color(speed[t]), -1, cv2.LINE_AA)

            # 5) 속도 표시
            v = speed[t]
            ov = frame.copy()
            cv2.rectangle(ov, (px(12), px(12)), (px(330), px(72)), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
            if np.isfinite(v):
                arrow = _direction(vel[t, 0], vel[t, 1], v)
                cv2.putText(frame, f"COG {v:4.2f} m/s {arrow}", (px(22), px(50)), cv2.FONT_HERSHEY_SIMPLEX,
                            fs(0.75), (255, 255, 255), px(2), cv2.LINE_AA)

            # 6) 하단 속도 그래프
            ov = frame.copy()
            cv2.rectangle(ov, (gx0 - px(40), gy0 - px(22)), (gx1 + px(10), gy1 + px(12)), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.40, frame, 0.60, 0, frame)
            step = 0.5 if vmax <= 3 else 1.0
            for lv in np.arange(step, vmax, step):
                y = vy(lv)
                cv2.line(frame, (gx0, y), (gx1, y), (120, 120, 120), px(1))
                cv2.putText(frame, f"{lv:.1f}", (gx0 - px(36), y + px(4)), cv2.FONT_HERSHEY_SIMPLEX, fs(0.4), (200, 200, 200), px(1), cv2.LINE_AA)
            for i in range(s + 1, e):
                if np.isfinite(speed[i - 1:i + 1]).all():
                    col = speed_color(speed[i]) if i <= t else (110, 110, 110)
                    cv2.line(frame, (tx(i - 1), vy(speed[i - 1])), (tx(i), vy(speed[i])), col, px(2), cv2.LINE_AA)
            for p in moves:
                if p <= t:
                    cv2.circle(frame, (tx(p), vy(speed[p])), px(3), (40, 150, 255), -1, cv2.LINE_AA)
            cv2.line(frame, (tx(t), gy0), (tx(t), gy1), (255, 255, 255), px(1))
            cv2.putText(frame, "COG speed (m/s)", (gx0, gy0 - px(6)), cv2.FONT_HERSHEY_SIMPLEX, fs(0.45), (230, 230, 230), px(1), cv2.LINE_AA)
        vw.write(frame)
        t += 1
        if progress and (t % 10 == 0):
            progress(t, n_total)
    vw.release()

    ff = ffmpeg_exe()
    if ff:
        subprocess.run([ff, "-loglevel", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", "-preset", "veryfast", "-movflags", "+faststart", out_path], check=True)
        os.remove(tmp)
    else:
        shutil.move(tmp, out_path)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("video")
    ap.add_argument("--height", type=float, default=175)
    ap.add_argument("--wingspan", type=float, default=178)
    ap.add_argument("--out", default="lesson.mp4")
    a = ap.parse_args()
    A = AN.analyze(np.load(a.raw), a.height, a.wingspan)
    render(A, a.video, a.out)
    print("saved", a.out)
