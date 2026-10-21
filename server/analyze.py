"""
3단계: 실제 단위(m, m/s) 분석

키와 윙스팬으로 픽셀→미터 스케일을 추정하고, 무게중심(COG) 속도를 계산한다.

스케일 추정 원리
  - 영상 속 팔·다리 길이는 카메라 쪽으로 기울면 짧아 보이지만, 길어 보이는 쪽에는 한계가 있다.
    등반 중 상위 5% 길이 ≈ 실제 길이(화면과 평행할 때)로 본다.
  - 실제 길이는 인체 비율(Drillis & Contini)로 계산
      어깨→손목 = 0.332 × 윙스팬,  엉덩이→발목 = 0.491 × 키
  - 카메라가 아래에서 올려다보면 벽 위쪽일수록 작게 보인다(원근).
    몸통 길이가 화면 높이(y)에 따라 어떻게 변하는지 직선으로 맞춰서 높이별 스케일을 보정한다.

보정 자세 (권장)
  - 등반 전에 벽에 등을 붙이고 카메라를 보며 양팔을 수평으로 쭉 펴고 1~2초 서 있기
  - 이 구간을 자동으로 찾아 손목 사이 거리로 스케일을 잡는다 (없으면 위 신체 비율 방식)

사용법: python analyze.py test.raw.npz --height 175 --wingspan 178
"""
import argparse
import numpy as np
from scipy.signal import savgol_filter, find_peaks

import refine as R


# ────────────────────────────────────────────────────────────────────────────
def climb_segment(xy, max_gap=12, size_ratio=1.5):
    """카메라 앞을 걸어다니는 구간을 빼고, 벽에 붙어 있는 가장 긴 연속 구간 [s, e)."""
    with np.errstate(all="ignore"):
        ext = np.nanmax(xy[:, 5:, 1], 1) - np.nanmin(xy[:, :, 1], 1)
    ok = np.isfinite(ext)
    med = np.nanmedian(ext[ok])
    on = ok & (ext < size_ratio * med)
    # 짧은 끊김은 이어 붙이고 가장 긴 구간 선택
    runs, t, T = [], 0, len(on)
    while t < T:
        if on[t]:
            s = t
            last = t
            while t < T and (on[t] or (t - last) <= max_gap):
                if on[t]:
                    last = t
                t += 1
            runs.append((s, last + 1))
        else:
            t += 1
    return max(runs, key=lambda r: r[1] - r[0])


def _len(xy, a, b):
    return np.linalg.norm(xy[:, a] - xy[:, b], axis=-1)


def perspective_fit(xy, seg):
    """몸통 길이(px) = a·y + b. 반환: y → 상대 배율 함수 (기준 = 구간 중앙 높이에서 1.0)."""
    s, e = seg
    torso = (_len(xy, 5, 11) + _len(xy, 6, 12))[s:e] / 2
    hy = np.nanmean(xy[s:e][:, [11, 12], 1], 1)
    ok = np.isfinite(torso) & np.isfinite(hy)
    a, b = np.polyfit(hy[ok], torso[ok], 1)
    y_ref = float(np.median(hy[ok]))
    ref = a * y_ref + b
    corr = float(np.corrcoef(hy[ok], torso[ok])[0, 1])
    # 너무 약한 상관이면 보정하지 않음
    if corr < 0.3:
        a = 0.0
    f = lambda y: np.clip((a * np.asarray(y) + b) / ref, 0.5, 2.0) if a else np.ones_like(np.asarray(y, float))
    return f, dict(slope=float(a), corr=corr, y_ref=y_ref)


def estimate_scale(xy, seg, height_m, wingspan_m, f):
    """기준 높이(y_ref)에서의 px/m 와, 팔·다리 각각의 추정치를 반환."""
    s, e = seg
    hy = np.nanmean(xy[s:e][:, [11, 12], 1], 1)
    fy = f(hy)
    arm = np.r_[(_len(xy, 5, 7) + _len(xy, 7, 9))[s:e] / fy, (_len(xy, 6, 8) + _len(xy, 8, 10))[s:e] / fy]
    leg = np.r_[(_len(xy, 11, 13) + _len(xy, 13, 15))[s:e] / fy, (_len(xy, 12, 14) + _len(xy, 14, 16))[s:e] / fy]
    arm_px = np.nanpercentile(arm, 95)
    leg_px = np.nanpercentile(leg, 95)
    s_arm = arm_px / (0.332 * wingspan_m)
    s_leg = leg_px / (0.491 * height_m)
    return float(np.mean([s_arm, s_leg])), dict(px_per_m_arm=float(s_arm), px_per_m_leg=float(s_leg))


def find_calibration(xy, fps, min_sec=0.7):
    """보정 자세 찾기: 카메라를 보고 양팔을 수평으로 쭉 편 채 서 있는 구간 (min_sec 이상).
    반환: (시작, 끝, 손목 사이 거리 px, 그때 어깨 높이 y) 또는 None"""
    sh_l, sh_r, el_l, el_r, wr_l, wr_r = (xy[:, i] for i in (5, 6, 7, 8, 9, 10))
    torso = (_len(xy, 5, 11) + _len(xy, 6, 12)) / 2
    with np.errstate(all="ignore"):
        arm_l = _len(xy, 5, 7) + _len(xy, 7, 9)
        arm_r = _len(xy, 6, 8) + _len(xy, 8, 10)
        straight = (_len(xy, 5, 9) > 0.9 * arm_l) & (_len(xy, 6, 10) > 0.9 * arm_r)
        level = (np.abs(wr_l[:, 1] - sh_l[:, 1]) < 0.25 * torso) & (np.abs(wr_r[:, 1] - sh_r[:, 1]) < 0.25 * torso)
        span = np.linalg.norm(wr_l - wr_r, axis=-1)
        sh_w = np.linalg.norm(sh_l - sh_r, axis=-1)
        spread = span > 2.5 * sh_w            # 손목이 어깨 폭보다 훨씬 바깥
        sym = np.abs(arm_l - arm_r) < 0.2 * np.maximum(arm_l, arm_r)   # 양팔 길이가 비슷 = 정면
    ok = straight & level & spread & sym
    best, t, T = None, 0, len(ok)
    while t < T:
        if ok[t]:
            s0 = t
            while t < T and ok[t]:
                t += 1
            if t - s0 >= min_sec * fps and (best is None or t - s0 > best[1] - best[0]):
                best = (s0, t)
        else:
            t += 1
    if best is None:
        return None
    s0, e0 = best
    # 팔을 가장 쭉 편 순간들(상위 30%)의 손목 거리
    sp = np.sort(span[s0:e0])[-max(1, (e0 - s0) * 3 // 10):]
    y = float(np.nanmedian((sh_l[s0:e0, 1] + sh_r[s0:e0, 1]) / 2))
    return s0, e0, float(np.median(sp)), y


def cog_kinematics(cog, seg, fps, px_per_m, f, win_s=0.45):
    """COG 속도(m/s), 수직 속도(위 +), 가속도 크기(m/s²). 구간 밖은 NaN."""
    s, e = seg
    T = len(cog)
    vel = np.full((T, 2), np.nan)
    acc = np.full((T, 2), np.nan)
    c = cog[s:e].copy()
    ok = np.isfinite(c[:, 0])
    if ok.sum() > 10:
        idx = np.arange(len(c))
        c[~ok] = np.c_[np.interp(idx[~ok], idx[ok], c[ok, 0]), np.interp(idx[~ok], idx[ok], c[ok, 1])]
        win = max(5, int(round(win_s * fps)) | 1)
        v_px = savgol_filter(c, win, 2, deriv=1, axis=0) * fps
        a_px = savgol_filter(c, win + 4, 2, deriv=2, axis=0) * fps * fps
        scale = (px_per_m * f(c[:, 1]))[:, None]   # 그 높이에서의 px/m
        vel[s:e] = v_px / scale
        acc[s:e] = a_px / scale
        vel[s:e][~ok] = np.nan
        acc[s:e][~ok] = np.nan
    vel[:, 1] *= -1   # 화면 y는 아래가 + → 위를 + 로
    acc[:, 1] *= -1
    speed = np.linalg.norm(vel, axis=1)
    return speed, vel, acc


def find_moves(speed, fps, min_speed=0.25):
    """속도 피크 = 한 동작. [(프레임, 최고속도)]"""
    sp = np.nan_to_num(speed)
    peaks, _ = find_peaks(sp, height=min_speed, distance=int(0.6 * fps), prominence=0.12)
    return [(int(p), float(sp[p])) for p in peaks]


def analyze(raw, height_cm, wingspan_cm):
    r = R.refine(raw)
    fps = float(raw["fps"])
    seg = climb_segment(r["xy"])
    f, pinfo = perspective_fit(r["xy"], seg)
    px_per_m, sinfo = estimate_scale(r["xy"], seg, height_cm / 100, wingspan_cm / 100, f)
    sinfo["method"] = "body_proportion"
    cal = find_calibration(r["xy"], fps)
    if cal is not None:
        # 손목~손목 = 윙스팬 - 손 길이 2개 (손 ≈ 윙스팬의 10.8%) → 윙스팬의 78.4%
        s_cal = cal[2] / (0.784 * wingspan_cm / 100)     # 보정 자세 높이에서의 px/m
        s_cal_ref = s_cal / float(f(cal[3]))              # 기준 높이로 환산
        sinfo.update(method="calibration_pose", calib_frames=(cal[0], cal[1]),
                     px_per_m_calib=float(s_cal_ref), px_per_m_proportion=px_per_m)
        px_per_m = s_cal_ref
    speed, vel, acc = cog_kinematics(r["cog"], seg, fps, px_per_m, f)
    moves = find_moves(speed, fps)
    return dict(refined=r, fps=fps, seg=seg, persp=pinfo, persp_f=f, px_per_m=px_per_m,
                scale_info=sinfo, speed=speed, vel=vel, acc=acc, moves=moves)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?", default="test.raw.npz")
    ap.add_argument("--height", type=float, default=175, help="키 (cm)")
    ap.add_argument("--wingspan", type=float, default=178, help="윙스팬 (cm)")
    a = ap.parse_args()
    A = analyze(np.load(a.raw), a.height, a.wingspan)
    fps = A["fps"]
    s, e = A["seg"]
    print(f"등반 구간: {s / fps:.1f}s ~ {e / fps:.1f}s")
    print(f"원근 보정: 상관 {A['persp']['corr']:.2f}, 기울기 {A['persp']['slope']:.3f}")
    si = A["scale_info"]
    if si["method"] == "calibration_pose":
        c0, c1 = si["calib_frames"]
        print(f"보정 자세 감지: {c0 / fps:.1f}s ~ {c1 / fps:.1f}s → {A['px_per_m']:.1f} px/m "
              f"(신체 비율 방식이면 {si['px_per_m_proportion']:.1f})")
    else:
        print("보정 자세 없음 → 신체 비율 방식 사용")
        print(f"스케일(기준 높이): {A['px_per_m']:.1f} px/m  (팔 기준 {si['px_per_m_arm']:.1f}, 다리 기준 {si['px_per_m_leg']:.1f})")
    sp = A["speed"]
    print(f"COG 평균 속도 {np.nanmean(sp):.2f} m/s, 최고 {np.nanmax(sp):.2f} m/s")
    print("동작별 최고 속도:")
    for p, v in A["moves"]:
        vx, vy = A["vel"][p]
        d = ("위" if vy > abs(vx) * 0.7 else "아래" if -vy > abs(vx) * 0.7 else "") + \
            ("" if abs(vx) < abs(vy) * 0.7 else ("오른쪽" if vx > 0 else "왼쪽"))
        print(f"  {p / fps:5.1f}s  {v:.2f} m/s  ({d})")
