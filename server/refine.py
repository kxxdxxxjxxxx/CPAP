"""
2단계: 오프라인 보정 엔진 (YOLO 불필요, numpy/scipy만 사용)

extract_raw.py 가 만든 .raw.npz 를 읽어서
  1) 클라이머 추적 (가장 큰 사람이 아니라, 직전 프레임과 이어지는 사람)
  2) 좌우 뒤바뀜 교정
  3) 오류 관절 검출 (신뢰도 대신 순간이동 + 뼈 길이로 판단)
  4) 접촉/이동 구간 분리 → 접촉 중엔 그 위치에 고정, 이동 중엔 양방향 보간
  5) 스무딩
  6) 분절 질량 모델(de Leva 1996)로 무게중심 계산
을 수행하고 결과를 .refined.npz 로 저장한다.

사용법: python refine.py test.raw.npz
"""
import sys
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sh", "r_sh", "l_el", "r_el",
         "l_wr", "r_wr", "l_hip", "r_hip", "l_kn", "r_kn", "l_an", "r_an"]
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
         (5, 6), (11, 12), (5, 11), (6, 12)]
EXTREMITIES = [9, 10, 15, 16]          # 손목, 발목 → 접촉 구간 분석 대상

P = dict(
    track_gate=0.6,           # 클라이머 추적: 직전 박스 대각선 대비 허용 이동 비율
    swap_ratio=0.7,           # 좌우 교정: 뒤바꾼 쪽 비용이 이 비율보다 작으면 교정
    bone_max=1.5,             # 뼈 길이가 중앙값의 몇 배를 넘으면 오류로 볼지
    jump_dev=0.45,            # 중앙값 필터 대비 이탈이 몸통 길이의 몇 배면 오류로 볼지
    contact_speed=0.05,       # 접촉 판정 속도 (몸통 길이 / 프레임)
    contact_min_frames=6,     # 접촉 최소 지속 프레임 (24fps 기준 0.25초)
    smooth_win=7,             # Savitzky-Golay 창 크기 (프레임)
    max_gap=36,               # 이보다 긴 미검출 구간은 보간하지 않음 (1.5초)
)


# ────────────────────────────────────────────────────────────────────────────
# 1. 클라이머 추적
# ────────────────────────────────────────────────────────────────────────────
def track_climber(kpts, boxes, count, seed=None):
    """프레임마다 직전 클라이머 박스와 가장 가까운 사람을 고른다. 앞뒤 양방향으로 전파.
    반환: (T,17,3) 키포인트 (없으면 NaN), (T,) 유효 여부"""
    T = len(count)
    if seed is None:  # 한 명만 잡힌 프레임 중 가운데
        single = np.where(count == 1)[0]
        seed = single[len(single) // 2]
    chosen = np.full(T, -1)
    chosen[seed] = 0

    def run(order):
        last_box, gap = boxes[seed, 0, :4], 0
        for t in order:
            best, best_d = -1, np.inf
            lc = (last_box[:2] + last_box[2:]) / 2
            diag = np.linalg.norm(last_box[2:] - last_box[:2])
            for i in range(count[t]):
                b = boxes[t, i, :4]
                d = np.linalg.norm((b[:2] + b[2:]) / 2 - lc)
                if d < best_d:
                    best, best_d = i, d
            if best >= 0 and best_d < diag * P["track_gate"] * (1 + 0.3 * gap):
                chosen[t], last_box, gap = best, boxes[t, best, :4], 0
            else:
                gap += 1

    run(range(seed + 1, T))
    run(range(seed - 1, -1, -1))
    out = np.full((T, 17, 3), np.nan, np.float64)
    ok = chosen >= 0
    out[ok] = kpts[np.where(ok)[0], chosen[ok]]
    return out, ok


# ────────────────────────────────────────────────────────────────────────────
# 2. 좌우 뒤바뀜 교정
# ────────────────────────────────────────────────────────────────────────────
def fix_lr_swaps(xy, valid, max_run=8):
    """팔/다리/몸통을 사슬 단위로 검사해 '잠깐 뒤바뀌었다가 돌아오는' 구간만 교정.

    직전 프레임만 보고 계속 교정하면, 한 번 잘못 뒤집힌 상태를 끝까지 끌고 가는 문제가 있다.
    그래서 원본에서 라벨이 뒤집히는 순간(flip)을 찾아 상태열을 만들고,
    max_run 프레임(24fps 기준 0.33초) 이하로 짧게 유지된 상태만 앞뒤 상태로 되돌린다.
    길게 유지된 상태는 모델 판단을 믿는다 (실제로 다리를 교차한 경우 등)."""
    xy = xy.copy()
    chains = {"arm": [(7, 8), (9, 10)], "leg": [(13, 14), (15, 16)],
              "body": [(5, 6), (11, 12)]}
    vt = np.where(valid)[0]
    n_fix = 0
    for pairs in chains.values():
        state = np.zeros(len(vt), int)
        for i in range(1, len(vt)):
            t0, t1 = vt[i - 1], vt[i]
            c_id = sum(np.linalg.norm(xy[t1, a] - xy[t0, a]) + np.linalg.norm(xy[t1, b] - xy[t0, b]) for a, b in pairs)
            c_sw = sum(np.linalg.norm(xy[t1, a] - xy[t0, b]) + np.linalg.norm(xy[t1, b] - xy[t0, a]) for a, b in pairs)
            flip = c_sw < P["swap_ratio"] * c_id
            state[i] = state[i - 1] ^ int(flip)
        # 짧은 상태 구간 → 이웃 상태로 되돌림
        fix = np.zeros(len(vt), bool)
        i = 0
        while i < len(vt):
            j = i
            while j < len(vt) and state[j] == state[i]:
                j += 1
            if i > 0 and j < len(vt) and (j - i) <= max_run:
                fix[i:j] = True
            i = j
        for i in np.where(fix)[0]:
            t = vt[i]
            for a, b in pairs:
                xy[t, [a, b]] = xy[t, [b, a]]
        n_fix += int(fix.sum())
    return xy, n_fix


# ────────────────────────────────────────────────────────────────────────────
# 3. 오류 관절 검출 (신뢰도 대신 운동학 기준)
# ────────────────────────────────────────────────────────────────────────────
def detect_bad(xy, valid, torso):
    bad = ~np.repeat(valid[:, None], 17, 1)
    # (a) 같은 쪽 두 관절이 한 점에 겹침 (두 손목이 한 손 위에 찍히는 경우)
    for a, b in [(9, 10), (15, 16), (7, 8), (13, 14)]:
        d = np.linalg.norm(xy[:, a] - xy[:, b], axis=-1)
        bad[:, a] |= d < 0.08 * torso
        bad[:, b] |= d < 0.08 * torso
    # (b) 뼈 길이 이상 → 말단 관절을 오류로
    #     카메라와의 거리에 따라 몸 크기가 변하므로, 그 프레임의 몸통 길이 대비 비율로 본다
    tl = np.linalg.norm((xy[:, 5] + xy[:, 6]) / 2 - (xy[:, 11] + xy[:, 12]) / 2, axis=-1)
    tl = median_filter(np.nan_to_num(tl, nan=torso), size=15, mode="nearest")
    for a, b in BONES[:8]:
        L = np.linalg.norm(xy[:, a] - xy[:, b], axis=-1) / tl
        med = np.nanmedian(L[valid])
        bad[:, b] |= L > P["bone_max"] * med
    # (c) 시간축 중앙값 필터에서 크게 벗어남 (순간이동)
    filled = fill_gaps(xy, bad, max_gap=10 ** 6)
    med = median_filter(filled, size=(9, 1, 1), mode="nearest")
    dev = np.linalg.norm(filled - med, axis=-1)
    bad |= dev > P["jump_dev"] * torso
    return bad


# ────────────────────────────────────────────────────────────────────────────
# 4. 보간 / 접촉 구간
# ────────────────────────────────────────────────────────────────────────────
def fill_gaps(xy, bad, max_gap=None):
    """오류/미검출 구간을 앞뒤 정상값으로 PCHIP 보간 (양방향 정보 사용)."""
    max_gap = P["max_gap"] if max_gap is None else max_gap
    out = xy.copy()
    T = len(xy)
    t_all = np.arange(T)
    for j in range(17):
        good = ~bad[:, j] & np.isfinite(xy[:, j, 0])
        if good.sum() < 2:
            continue
        tg = t_all[good]
        for c in range(2):
            f = PchipInterpolator(tg, xy[good, j, c], extrapolate=False)
            v = f(t_all)
            # 너무 긴 구간은 비워둔다
            gap_len = np.zeros(T)
            idx = np.searchsorted(tg, t_all)
            prev_t = np.where(idx > 0, tg[np.clip(idx - 1, 0, len(tg) - 1)], -10 ** 6)
            next_t = np.where(idx < len(tg), tg[np.clip(idx, 0, len(tg) - 1)], 10 ** 6)
            gap_len = next_t - prev_t
            v[(~good) & (gap_len > max_gap)] = np.nan
            out[:, j, c] = np.where(good, xy[:, j, c], v)
    return out


def smooth_runs(xy, win, keep=None):
    """관절별로 값이 연속으로 있는 구간마다 Savitzky-Golay 스무딩. keep=True 인 곳은 원래 값 유지."""
    out = xy.copy()
    T, J = xy.shape[:2]
    for j in range(J):
        ok = np.isfinite(xy[:, j, 0])
        t = 0
        while t < T:
            if not ok[t]:
                t += 1
                continue
            s = t
            while t < T and ok[t]:
                t += 1
            if t - s >= win:
                sm = savgol_filter(xy[s:t, j], win, 2, axis=0)
                if keep is not None:
                    sm = np.where(keep[s:t, j, None], xy[s:t, j], sm)
                out[s:t, j] = sm
    return out


def find_contacts(xy, torso):
    """손목/발목의 접촉 구간 [(관절, 시작, 끝, 고정위치)]."""
    contacts = []
    for j in EXTREMITIES:
        p = xy[:, j]
        ok = np.isfinite(p[:, 0])
        if ok.sum() < 10:
            continue
        ps = smooth_runs(p[:, None], 9)[:, 0]
        sp = np.r_[np.inf, np.linalg.norm(np.diff(ps, axis=0), axis=-1)]
        still = (sp < P["contact_speed"] * torso) & ok
        t = 0
        while t < len(still):
            if still[t]:
                s = t
                while t < len(still) and still[t]:
                    t += 1
                if t - s >= P["contact_min_frames"]:
                    contacts.append((j, s, t, np.median(p[s:t], axis=0)))
            else:
                t += 1
    return contacts


# ────────────────────────────────────────────────────────────────────────────
# 5. 무게중심: 분절 질량 모델 (de Leva 1996, 남성 값)
# ────────────────────────────────────────────────────────────────────────────
# (질량비, 근위점, 원위점, 근위점에서 무게중심까지 비율)
def segment_cog(xy):
    sh = (xy[:, 5] + xy[:, 6]) / 2
    hip = (xy[:, 11] + xy[:, 12]) / 2
    with np.errstate(all="ignore"):
        head = np.nanmean(xy[:, [0, 3, 4]], axis=1)
    segs = [
        (0.0694, head, head, 0.0),                       # 머리
        (0.4346, sh, hip, 0.4486),                       # 몸통
    ]
    for s, e, w in [(5, 7, 9), (6, 8, 10)]:
        segs += [(0.0271, xy[:, s], xy[:, e], 0.5772),   # 위팔
                 (0.0162, xy[:, e], xy[:, w], 0.4574),   # 아래팔
                 (0.0061, xy[:, w], xy[:, w] + 0.35 * (xy[:, w] - xy[:, e]), 0.5)]  # 손
    for h, k, a in [(11, 13, 15), (12, 14, 16)]:
        segs += [(0.1416, xy[:, h], xy[:, k], 0.4095),   # 허벅지
                 (0.0433, xy[:, k], xy[:, a], 0.4459),   # 정강이
                 (0.0137, xy[:, a], xy[:, a], 0.0)]      # 발
    tot = sum(s[0] for s in segs)
    return sum(m * (p + r * (d - p)) for m, p, d, r in segs) / tot


# ────────────────────────────────────────────────────────────────────────────
def refine(raw):
    kpts, boxes, count = raw["kpts"], raw["boxes"], raw["count"]
    k, valid = track_climber(kpts, boxes, count)
    xy = k[..., :2]
    torso = np.nanmedian(np.linalg.norm((xy[valid, 5] + xy[valid, 6]) / 2 - (xy[valid, 11] + xy[valid, 12]) / 2, axis=-1))

    xy, n_swap = fix_lr_swaps(xy, valid)
    bad = detect_bad(xy, valid, torso)
    xy_f = fill_gaps(xy, bad)

    # 접촉 구간: 그 구간 안의 값은 고정 위치로 (가려져도 손은 홀드에 있음)
    contacts = find_contacts(xy_f, torso)
    in_contact = np.zeros(bad.shape, bool)
    for j, s, e, pos in contacts:
        xy_f[s:e, j] = pos
        in_contact[s:e, j] = True

    # 이동 구간 스무딩 (접촉 구간은 이미 고정). 값이 있는 연속 구간에만 적용
    out = smooth_runs(xy_f, P["smooth_win"], keep=in_contact)
    # 접촉 경계가 툭 끊기지 않게 손발만 짧게 한 번 더
    out[:, EXTREMITIES] = smooth_runs(out[:, EXTREMITIES], 5)

    cog = segment_cog(out)
    stats = dict(torso_px=float(torso), lr_swaps_fixed=int(n_swap),
                 bad_joint_frames=int(bad[valid].sum()), contacts=len(contacts),
                 valid_frames=int(valid.sum()))
    return dict(xy=out, cog=cog, bad=bad, valid=valid, in_contact=in_contact,
                contacts=contacts, torso=torso, stats=stats, raw_xy=k[..., :2])


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test.raw.npz"
    raw = np.load(path)
    r = refine(raw)
    print(r["stats"])
    np.savez_compressed(path.replace(".raw.npz", ".refined.npz"),
                        xy=r["xy"], cog=r["cog"], bad=r["bad"], valid=r["valid"],
                        in_contact=r["in_contact"], fps=raw["fps"], size=raw["size"])
