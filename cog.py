import math
import numpy as np

COG_WEIGHTS = np.array([
    0.07,  # nose
    0.00,  # left_eye
    0.00,  # right_eye
    0.00,  # left_ear
    0.00,  # right_ear
    0.10, 0.10, 0.08, 0.08, 0.06, 0.06,  # 어깨, 팔꿈치, 손목
    0.12, 0.12, 0.08, 0.08, 0.05, 0.05   # 골반, 무릎, 발목
])


def calc_cog(kpts, offset_ratio=0.1):
    """COG(Center of Gravity)를 17 키포인트에서 가중평균으로 추정.
    상체가 더 무거우므로 골반 아래 방향으로 약간 오프셋.
    """
    w = COG_WEIGHTS / COG_WEIGHTS.sum()
    cog = (kpts * w[:, None]).sum(axis=0)

    shoulder_mid = (kpts[5] + kpts[6]) / 2.0
    hip_mid = (kpts[11] + kpts[12]) / 2.0
    torso_len = abs(hip_mid[1] - shoulder_mid[1])

    cog[1] += torso_len * offset_ratio
    return cog


def smooth_points(curr, prev, alpha=0.3):
    """레거시 EMA 스무딩 - 단순 호환용. 신규 코드는 KeypointStabilizer 사용 권장."""
    if prev is None:
        return curr
    return alpha * curr + (1 - alpha) * prev


# ────────────────────────────────────────────────────────────────────────────
# One Euro Filter & Keypoint Stabilizer
# ────────────────────────────────────────────────────────────────────────────
class OneEuroFilter:
    """1ε 필터 (One Euro Filter): 적응형 저역 필터.

    Casiez et al. 2012 "1€ Filter: A Simple Speed-based Low-pass Filter
    for Noisy Input in Interactive Systems" 구현.

    - 정지 상태(작은 |dx|): cutoff ≈ mincutoff → 강한 저역(부드럽게)
    - 빠른 움직임(큰 |dx|): cutoff = mincutoff + beta·|dx| → 즉각 추종(lag 적음)
    """

    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0,
                 beta: float = 0.0, dcutoff: float = 1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / max(self.freq, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = float(x)
            return float(x)
        dx = (float(x) - self.x_prev) * self.freq
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * float(x) + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0


class KeypointStabilizer:
    """포즈 17개 키포인트 시간적 안정화.

    1) One Euro Filter (x, y 각각, 키포인트 17개 → 총 34개 필터)로 지터 제거
    2) 속도 게이트: 한 프레임에 영상 대각선의 max_speed_ratio 이상 점프하면 거절 → 등속 예측 사용
    3) 미검출/저신뢰 폴백: 마지막 위치 + 마지막 속도로 외삽하여 max_predict_frames 만큼 유지
    4) (옵션) 손목(9, 10)이 가려진 경우: 예측 위치가 hold 박스 *안*이거나 매우 근접할 때만 그 hold 중심으로 보정
       발목은 스미어링·교차 발 문제로 hold-snap 비활성화

    이전 파이프라인의 estimate_occluded_keypoints + smooth_points 를 통합 대체.
    """
    NUM_KPTS = 17

    def __init__(
        self,
        fps: float = 30.0,
        mincutoff: float = 1.0,
        beta: float = 0.05,
        max_speed_ratio: float = 0.12,
        max_predict_frames: int = 10,
        conf_threshold: float = 0.30,
        wrist_hold_snap_radius_ratio: float = 0.04,
        snap_only_body_occluded: bool = True,
        body_occlusion_overlap_threshold: float = 0.4,
    ):
        self.fps = float(fps)
        self.filters_x = [OneEuroFilter(freq=fps, mincutoff=mincutoff, beta=beta) for _ in range(self.NUM_KPTS)]
        self.filters_y = [OneEuroFilter(freq=fps, mincutoff=mincutoff, beta=beta) for _ in range(self.NUM_KPTS)]
        self.last_pos = [None] * self.NUM_KPTS
        self.last_vel = [np.zeros(2, dtype=np.float64) for _ in range(self.NUM_KPTS)]
        self.miss_count = [0] * self.NUM_KPTS
        self.max_speed_ratio = float(max_speed_ratio)
        self.max_predict_frames = int(max_predict_frames)
        self.conf_threshold = float(conf_threshold)
        self.wrist_hold_snap_radius_ratio = float(wrist_hold_snap_radius_ratio)
        # 손목 hold-snap 을 "몸에 가려진 홀드"로만 한정하는 모드
        # (잘 보이는 홀드로 손이 점프해 가는 잘못된 추정 방지)
        self.snap_only_body_occluded = bool(snap_only_body_occluded)
        self.body_occlusion_overlap_threshold = float(body_occlusion_overlap_threshold)
        self.frame_diag = 1500.0

    def set_frame_size(self, w: int, h: int):
        self.frame_diag = float(np.sqrt(float(w) * w + float(h) * h))

    @property
    def max_step_px(self) -> float:
        return self.frame_diag * self.max_speed_ratio

    @property
    def wrist_snap_radius_px(self) -> float:
        return self.frame_diag * self.wrist_hold_snap_radius_ratio

    def _build_body_polygon(self, kpts_xyc: np.ndarray) -> "np.ndarray | None":
        """현재 raw 키포인트에서 어깨(5,6)/엉덩이(11,12)로 4각형을 만든다.
        해당 점이 저신뢰일 경우 last_pos 에서 보충한다. 4점 모두 확보 못하면 None.
        반환 순서: [L_shoulder, R_shoulder, R_hip, L_hip] (시계방향).
        """
        order = [5, 6, 12, 11]
        pts = []
        for i in order:
            if (
                float(kpts_xyc[i, 2]) >= self.conf_threshold
                and not (float(kpts_xyc[i, 0]) == 0.0 and float(kpts_xyc[i, 1]) == 0.0)
            ):
                pts.append(np.array([float(kpts_xyc[i, 0]), float(kpts_xyc[i, 1])], dtype=np.float64))
            elif self.last_pos[i] is not None:
                pts.append(self.last_pos[i].astype(np.float64))
            else:
                return None
        return np.array(pts, dtype=np.float64)

    @staticmethod
    def _point_in_polygon(p, poly) -> bool:
        """레이 캐스팅 point-in-polygon. poly: (N, 2)."""
        x, y = float(p[0]), float(p[1])
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = float(poly[i, 0]), float(poly[i, 1])
            xj, yj = float(poly[j, 0]), float(poly[j, 1])
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-9) + xi
            ):
                inside = not inside
            j = i
        return inside

    def _hold_body_overlap_ratio(self, hold_box, body_poly) -> float:
        """hold 박스가 body polygon 안에 얼마나 들어있는지 0~1 비율로 근사한다.
        4 모서리 + 중심 5점 샘플 중 polygon 안에 있는 비율. 가벼운 근사로 충분.
        """
        x1, y1, x2, y2 = float(hold_box[0]), float(hold_box[1]), float(hold_box[2]), float(hold_box[3])
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        samples = ((x1, y1), (x2, y1), (x2, y2), (x1, y2), (cx, cy))
        inside = sum(1 for p in samples if self._point_in_polygon(p, body_poly))
        return inside / float(len(samples))

    def _filter_body_occluded_holds(self, holds, body_poly):
        """body polygon 과 overlap_threshold 이상 겹치는 hold 만 반환."""
        if holds is None or len(holds) == 0 or body_poly is None:
            return []
        thr = self.body_occlusion_overlap_threshold
        out = []
        for h in holds:
            if self._hold_body_overlap_ratio(h, body_poly) >= thr:
                out.append(h)
        return out

    def _try_snap_to_hold(self, predicted: np.ndarray, holds) -> "np.ndarray | None":
        """예측 위치가 hold 박스 안이거나 wrist_snap_radius 이내일 때만 그 중심을 반환."""
        if holds is None or len(holds) == 0:
            return None
        best = None
        best_d = float("inf")
        # 1순위: 예측 위치를 포함하는 박스
        for h in holds:
            x1, y1, x2, y2 = float(h[0]), float(h[1]), float(h[2]), float(h[3])
            if x1 <= predicted[0] <= x2 and y1 <= predicted[1] <= y2:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                d = float(np.linalg.norm(np.array([cx, cy]) - predicted))
                if d < best_d:
                    best = np.array([cx, cy])
                    best_d = d
        if best is not None:
            return best
        # 2순위: 매우 근접한 박스 (반경 안)
        snap_r = self.wrist_snap_radius_px
        for h in holds:
            x1, y1, x2, y2 = float(h[0]), float(h[1]), float(h[2]), float(h[3])
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            d = float(np.linalg.norm(np.array([cx, cy]) - predicted))
            if d < snap_r and d < best_d:
                best = np.array([cx, cy])
                best_d = d
        return best

    def update(self, kpts_xyc: np.ndarray, holds=None,
               hold_snap_indices=(9, 10)) -> np.ndarray:
        """
        kpts_xyc: shape (17, 3) → [x, y, conf]
        holds: [[x1,y1,x2,y2], ...] 또는 None
        hold_snap_indices: 가려졌을 때 hold-snap 보정을 적용할 인덱스 (기본=손목 9,10)

        Returns: 안정화된 (17, 3). x, y는 필터링/예측된 값. conf는 폴백 시 0.4 로 마킹.
        """
        out = kpts_xyc.copy().astype(np.float64)
        max_step = self.max_step_px

        # 손목 hold-snap 후보 결정:
        # - snap_only_body_occluded=True: 몸(어깨-엉덩이 4각형)에 가려진 홀드만 후보로 사용
        #   → 잘 보이는 홀드로 손이 잘못 점프하는 현상 방지
        # - False: 기존 동작(모든 홀드)
        if self.snap_only_body_occluded:
            body_poly = self._build_body_polygon(kpts_xyc)
            snap_holds = self._filter_body_occluded_holds(holds, body_poly) if body_poly is not None else []
        else:
            snap_holds = holds

        for i in range(self.NUM_KPTS):
            raw = np.array([out[i, 0], out[i, 1]], dtype=np.float64)
            conf = float(out[i, 2])

            # 검출이 유효한지 판정
            valid_detection = (
                conf >= self.conf_threshold
                and not (raw[0] == 0.0 and raw[1] == 0.0)
            )

            use_obs: "np.ndarray | None" = None
            fallback_used = False

            if valid_detection and self.last_pos[i] is not None:
                step = float(np.linalg.norm(raw - self.last_pos[i]))
                if step > max_step:
                    # 속도 게이트: 점프 거절
                    valid_detection = False
                else:
                    use_obs = raw
            elif valid_detection:
                # 첫 검출
                use_obs = raw

            if not valid_detection:
                # 미검출/저신뢰/거절 → 등속 예측으로 폴백
                if self.last_pos[i] is not None and self.miss_count[i] < self.max_predict_frames:
                    pred = self.last_pos[i] + self.last_vel[i]
                    # 손목만: 예측 위치가 hold 안/근접일 때 그 중심으로 보정 (잘못된 hold 스냅 방지)
                    # snap_only_body_occluded=True 면 snap_holds 는 몸에 가려진 hold만.
                    if i in hold_snap_indices and snap_holds:
                        snap = self._try_snap_to_hold(pred, snap_holds)
                        if snap is not None:
                            pred = snap
                    use_obs = pred
                    self.miss_count[i] += 1
                    fallback_used = True
                else:
                    # 너무 오래 미검출 → 마지막 위치 유지 (또는 raw)
                    use_obs = self.last_pos[i] if self.last_pos[i] is not None else raw
                    self.miss_count[i] = self.max_predict_frames + 1
                    fallback_used = True

            # One Euro 필터 적용
            fx = self.filters_x[i].filter(float(use_obs[0]))
            fy = self.filters_y[i].filter(float(use_obs[1]))
            filtered = np.array([fx, fy], dtype=np.float64)

            # 속도/위치 갱신 (속도는 클립)
            if self.last_pos[i] is not None:
                vel = filtered - self.last_pos[i]
                vmag = float(np.linalg.norm(vel))
                if vmag > max_step:
                    vel = vel * (max_step / max(vmag, 1e-6))
                self.last_vel[i] = vel
            self.last_pos[i] = filtered

            if valid_detection and not fallback_used:
                self.miss_count[i] = 0

            out[i, 0] = filtered[0]
            out[i, 1] = filtered[1]
            if fallback_used:
                # 폴백 사용 시 표시용 conf (그리기 측에서 활용 가능)
                out[i, 2] = max(0.0, min(conf, 0.4)) if conf > 0 else 0.4

        return out
