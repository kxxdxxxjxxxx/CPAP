import os
import shutil
import subprocess
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from cog import calc_cog, smooth_points, KeypointStabilizer


def _get_ffmpeg_exe():
    """ffmpeg 실행 파일 경로를 반환. 시스템 ffmpeg 우선, 없으면 imageio-ffmpeg 번들 사용."""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _transcode_to_web_h264(src_path, dst_path):
    """
    OpenCV가 작성한 영상을 브라우저 호환 H.264(yuv420p, +faststart) MP4로 재인코딩.
    실패 시 src_path를 그대로 dst_path로 복사 (최후의 fallback).
    """
    ffmpeg = _get_ffmpeg_exe()
    if ffmpeg is None:
        # ffmpeg 사용 불가 - 원본을 그대로 사용
        if src_path != dst_path:
            shutil.move(src_path, dst_path)
        print("[pipeline] WARNING: ffmpeg을 찾을 수 없어 재인코딩을 건너뜁니다. 브라우저 재생이 안될 수 있습니다.")
        return

    cmd = [
        ffmpeg,
        "-y",
        "-i", src_path,
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-movflags", "+faststart",
        "-an",
        dst_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # 변환 성공 시 원본 임시파일 제거
        if os.path.exists(src_path) and src_path != dst_path:
            try:
                os.remove(src_path)
            except OSError:
                pass
        print(f"[pipeline] H.264 재인코딩 완료: {dst_path}")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
        print(f"[pipeline] ffmpeg 재인코딩 실패: {stderr[:500]}")
        # 실패 시 원본을 dst로 이동
        if src_path != dst_path and os.path.exists(src_path):
            shutil.move(src_path, dst_path)
    except Exception as e:
        print(f"[pipeline] ffmpeg 호출 오류: {e}")
        if src_path != dst_path and os.path.exists(src_path):
            shutil.move(src_path, dst_path)


# ────────────────────────────────────────────────────────────────────────────
# 런타임 튜닝 가능한 파라미터 (app.py의 /params API로 동적 변경됨)
# ────────────────────────────────────────────────────────────────────────────
OCCLUSION_THRESHOLD = 0.3        # (legacy) 키포인트 신뢰도 임계 — Stabilizer.conf_threshold 와 동일 의미
OCCLUSION_SEARCH_RADIUS = 150    # (legacy) 사용 안 함 — 현재는 wrist_hold_snap_radius_ratio 사용
SMOOTHING_ALPHA = 0.5            # (legacy) 사용 안 함 — One Euro 필터로 대체
HOLD_CONF = 0.1
POSE_CONF = 0.35
ARM_EXTENSION = 0.15
LEG_EXTENSION = 0.10

# Phase 1 신규 파라미터: One Euro Filter + 속도 게이트
ONE_EURO_MINCUTOFF = 1.0          # 정지 시 cutoff (낮을수록 더 부드러움)
ONE_EURO_BETA = 0.05              # 속도 비례 cutoff 가중 (높을수록 빠른 동작에서 lag 적음)
MAX_SPEED_RATIO = 0.12            # 한 프레임 최대 이동량 / 영상 대각선
MAX_PREDICT_FRAMES = 10           # 가려진 동안 등속 예측 유지 프레임 수
WRIST_HOLD_SNAP_RATIO = 0.04      # 손목 hold-snap 반경 / 영상 대각선
# 손목 hold-snap 을 "몸에 가려진 홀드"로만 한정 (잘 보이는 홀드로의 잘못된 점프 방지)
SNAP_ONLY_BODY_OCCLUDED = True
BODY_OCCLUSION_OVERLAP_THRESHOLD = 0.4  # hold 박스가 몸 4각형에 0~1 비율로 얼마나 겹쳐야 가려진 것으로 볼지
USE_HOLD_MODEL_PATH_DEFAULT = "C:/Project/runs/detect/train11/weights/best.pt"

# 1. 모델 로드 (포즈 모델 및 학습한 홀드 모델)
pose_model = YOLO("yolo11l-pose.pt")
_hold_model_path = USE_HOLD_MODEL_PATH_DEFAULT
hold_model = YOLO(_hold_model_path) if os.path.exists(_hold_model_path) else None

# 색상 설정 [cite: 1]
COLOR_LEFT, COLOR_RIGHT = (0, 140, 255), (255, 255, 0) 
COLOR_TORSO = (255, 120, 0)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (95, 95, 241)
COLOR_WHITE = (255, 255, 255)

def estimate_occluded_keypoints(kpts, holds, prev_kpts, threshold=None):
    """
    가려진 관절 좌표를 가장 인접한 홀드 좌표로 추정/보정하는 함수
    kpts: 현재 관절 [17, 3] (x, y, conf)
    holds: 탐지된 홀드들의 리스트 [[x1, y1, x2, y2], ...]
    prev_kpts: 이전 프레임 관절 [17, 2]
    """
    if threshold is None:
        threshold = OCCLUSION_THRESHOLD
    search_radius = OCCLUSION_SEARCH_RADIUS

    for i in [9, 10, 15, 16]:
        conf = kpts[i][2]
        if conf < threshold and prev_kpts is not None:
            prev_pos = prev_kpts[i]
            best_hold_center = prev_pos
            min_dist = float('inf')
            for h in holds:
                hold_center = np.array([(h[0]+h[2])/2, (h[1]+h[3])/2])
                dist = np.linalg.norm(prev_pos - hold_center)
                if dist < min_dist and dist < search_radius:
                    min_dist = dist
                    best_hold_center = hold_center
            kpts[i][0], kpts[i][1] = best_hold_center
            kpts[i][2] = 0.5
    return kpts

def draw_elliptical_limb(img, p1, p2, color):
    """타원형 팔다리 그리기 기능 [cite: 1, 2]"""
    p1, p2 = np.array(p1), np.array(p2)
    vec = p2 - p1
    dist = np.linalg.norm(vec)
    if dist < 5: return
    center = tuple(((p1 + p2) / 2).astype(int))
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    axes = (int(dist / 2), int(dist * 0.08))
    overlay = img.copy()
    cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    cv2.ellipse(img, center, axes, angle, 0, 360, COLOR_WHITE, 1, cv2.LINE_AA)

def draw_old_cog_marker(img, cog):
    """무게중심 마커 그리기 기능 [cite: 3]"""
    x, y = int(cog[0]), int(cog[1])
    cv2.circle(img, (x, y), 10, COLOR_WHITE, 2, cv2.LINE_AA)
    cv2.line(img, (x-15, y), (x+15, y), COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.line(img, (x, y-15), (x, y+15), COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 3, COLOR_RED, -1, cv2.LINE_AA)

def get_head_center(kpts, scores):
    """
    코와 귀 관절을 이용해 머리의 중앙을 추정하는 함수
    indices: 0(Nose), 3(L-Ear), 4(R-Ear)
    """
    # 귀(3, 4)가 둘 다 잘 보일 경우 귀의 중점을 우선 사용
    if scores[3] > 0.5 and scores[4] > 0.5:
        return (kpts[3] + kpts[4]) / 2
    
    # 한쪽 귀만 보일 경우 코와 보이는 귀의 중간 지점 활용
    visible_indices = [i for i in [0, 3, 4] if scores[i] > 0.3]
    if not visible_indices:
        return kpts[0] # 아무것도 안 보이면 코 유지
        
    return np.mean(kpts[visible_indices], axis=0)

def draw_enhanced_pose(img, kpts, display_option):
    # kpts: [17, 3] (x, y, scores) 가정
    coords = kpts[:, :2]
    scores = kpts[:, 2]
    
    if display_option == "no_ears":
        # ✅ 새롭게 계산된 머리 중앙점
        head_center = get_head_center(coords, scores)
        
        # 1) 몸통 사각형 (기존 유지)
        torso_pts = np.array([coords[5], coords[6], coords[12], coords[11]], np.int32)
        
        # 2) ✅ 수정된 상체 삼각형 (코 대신 계산된 head_center 사용)
        upper_pts = np.array([head_center, coords[5], coords[6]], np.int32)
        
        overlay = img.copy()
        cv2.fillPoly(overlay, [torso_pts], COLOR_TORSO)
        cv2.fillPoly(overlay, [upper_pts], COLOR_TORSO)
        
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        cv2.polylines(img, [torso_pts], True, COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.polylines(img, [upper_pts], True, COLOR_WHITE, 1, cv2.LINE_AA)
        
        # 몸통 및 상체 색상 채우기 (COLOR_TORSO)
        cv2.fillPoly(overlay, [torso_pts], COLOR_TORSO)
        cv2.fillPoly(overlay, [upper_pts], COLOR_TORSO)
        
        # 40% 투명도 적용
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        # 흰색 외곽선 그리기 (사각형 + 삼각형)
        cv2.polylines(img, [torso_pts], True, COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.polylines(img, [upper_pts], True, COLOR_WHITE, 1, cv2.LINE_AA)

        # 3) 팔다리 타원형 그리기 (연장 로직 반영됨)
        limbs = [(5,7), (7,9), (11,13), (13,15), (6,8), (8,10), (12,14), (14,16)]
        for s, e in limbs:
            col = COLOR_LEFT if s % 2 != 0 else COLOR_RIGHT
            draw_elliptical_limb(img, kpts[s][:2], kpts[e][:2], col)

    # 4) 관절 점 (반투명 도넛 형태, 두께 2)
    point_overlay = img.copy()
    target_indices = [9, 10, 15, 16] if display_option == "tips" else list(range(5, 17))
    for i in target_indices:
        cv2.circle(point_overlay, tuple(kpts[i][:2].astype(int)), 5, COLOR_WHITE, 1, cv2.LINE_AA)
    
    cv2.addWeighted(point_overlay, 0.7, img, 0.3, 0, img)

def get_tiled_holds(model, img, conf=None):
    """
    이미지를 4등분(15% 중첩)하여 작은 홀드들을 정밀하게 탐지하는 함수
    """
    if conf is None:
        conf = HOLD_CONF
    h, w = img.shape[:2]
    all_boxes = []

    # 1. 4개 구역 정의 (좌상, 우상, 좌하, 우하) + 15% 중첩 구역 포함
    # 중첩을 두어야 경계면에 걸친 홀드가 잘리지 않습니다.
    overlap = 0.15
    mid_w, mid_h = w // 2, h // 2
    
    # 각 타일의 (x_start, y_start, x_end, y_end)
    tiles = [
        (0, 0, int(mid_w * (1 + overlap)), int(mid_h * (1 + overlap))), # TL
        (int(mid_w * (1 - overlap)), 0, w, int(mid_h * (1 + overlap))), # TR
        (0, int(mid_h * (1 - overlap)), int(mid_w * (1 + overlap)), h), # BL
        (int(mid_w * (1 - overlap)), int(mid_h * (1 - overlap)), w, h)  # BR
    ]

    for (x1, y1, x2, y2) in tiles:
        tile_img = img[y1:y2, x1:x2]
        results = model(tile_img, conf=conf, verbose=False)
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            
            for box, score in zip(boxes, scores):
                # ✅ 타일 좌표를 전체 이미지 좌표로 변환
                global_box = [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1, score]
                all_boxes.append(global_box)

    if not all_boxes:
        return np.array([])

    # 2. 중복 제거 (NMS - Non Maximum Suppression)
    all_boxes = np.array(all_boxes)
    # 간단한 NMS: 겹치는 영역이 40% 이상이면 점수가 낮은 박스 제거
    keep = cv2.dnn.NMSBoxes(
        all_boxes[:, :4].tolist(), 
        all_boxes[:, 4].tolist(), 
        score_threshold=conf, 
        nms_threshold=0.4
    )
    
    return all_boxes[keep][:, :4] if len(keep) > 0 else np.array([])

def process_video(input_path, output_path, display_option="no_ears"):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 30.0

    # OpenCV는 mp4v(MPEG-4 Part 2) 코덱을 안정적으로 지원하지만 브라우저 호환성이 없음.
    # 우선 임시 파일에 mp4v로 기록한 뒤, ffmpeg으로 H.264(yuv420p, +faststart) 재인코딩하여
    # 브라우저(<video> 태그) 재생이 가능하도록 변환한다.
    base, ext = os.path.splitext(output_path)
    temp_path = base + "_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
    if not out.isOpened():
        # 일부 환경에서 mp4v 컨테이너 매칭이 실패하면 .avi/MJPG로 폴백
        temp_path = base + "_raw.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))

    cog_history = deque(maxlen=90)

    # Phase 1: 키포인트 안정화기 (One Euro Filter + 속도 게이트 + 가려짐 폴백)
    stabilizer = KeypointStabilizer(
        fps=fps,
        mincutoff=ONE_EURO_MINCUTOFF,
        beta=ONE_EURO_BETA,
        max_speed_ratio=MAX_SPEED_RATIO,
        max_predict_frames=MAX_PREDICT_FRAMES,
        conf_threshold=OCCLUSION_THRESHOLD,
        wrist_hold_snap_radius_ratio=WRIST_HOLD_SNAP_RATIO,
        snap_only_body_occluded=SNAP_ONLY_BODY_OCCLUDED,
        body_occlusion_overlap_threshold=BODY_OCCLUSION_OVERLAP_THRESHOLD,
    )
    stabilizer.set_frame_size(w, h)

    # 1. 정적 홀드 탐지 및 스크린샷 저장 (Tiling 적용)
    ret, first_frame = cap.read()
    if not ret: return

    detected_holds = get_tiled_holds(hold_model, first_frame) if hold_model else []
    
    # 홀드 지도 스크린샷 저장 로직
    screenshot_path = output_path.replace(".mp4", "_holds.jpg")
    screenshot = first_frame.copy()
    for h_box in detected_holds:
        cv2.rectangle(screenshot, (int(h_box[0]), int(h_box[1])), 
                     (int(h_box[2]), int(h_box[3])), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(screenshot_path, screenshot)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 다시 처음으로

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. 포즈 탐지
        pose_results = pose_model(frame, stream=True, conf=POSE_CONF, verbose=False)
        
        for res in pose_results:
            if res.keypoints is not None and len(res.keypoints) > 0:
                # 가장 큰 객체(클라이머) 선택 로직 [cite: 1]
                idx = res.boxes.xywh.cpu().numpy()[:, 2:4].prod(axis=1).argmax()
                
                # kpts shape: [17, 3] (x, y, confidence)
                kpts = res.keypoints.data[idx].cpu().numpy()

                # 2~3. 통합 안정화: One Euro 필터 + 속도 게이트 + 가려짐 폴백
                #   - 손목(9, 10)은 가려졌을 때 예측 위치가 hold 박스 안/근접일 때만 hold 중심으로 보정
                #   - 발목(15, 16)은 hold-snap 비활성화 (스미어링/발 교차 문제 방지)
                kpts = stabilizer.update(
                    kpts,
                    holds=detected_holds,
                    hold_snap_indices=(9, 10),
                )
                current_coords = kpts[:, :2]

                # 4. 무게중심(COG) 계산 및 시각화
                cog = calc_cog(current_coords)
                
                # 팔/다리 끝부분 개별 확장 로직
                for i, j in [(9, 7), (10, 8)]:
                    kpts[i][:2] = kpts[i][:2] + (kpts[i][:2] - kpts[j][:2]) * ARM_EXTENSION
                for i, j in [(15, 13), (16, 14)]:
                    kpts[i][:2] = kpts[i][:2] + (kpts[i][:2] - kpts[j][:2]) * LEG_EXTENSION

                # 5. COG 이동 경로(Trace) 그리기 [cite: 1]
                cog_history.append(tuple(cog.astype(int)))
                trace_overlay = frame.copy()
                for i in range(1, len(cog_history)):
                    cv2.line(trace_overlay, cog_history[i-1], cog_history[i], COLOR_WHITE, 6, cv2.LINE_AA)
                cv2.addWeighted(trace_overlay, 0.4, frame, 0.6, 0, frame)

                # 6. 최종 포즈 및 COG 마커 그리기 [cite: 1, 3]
                draw_enhanced_pose(frame, kpts, display_option)
                draw_old_cog_marker(frame, cog)

        out.write(frame)

    cap.release()
    out.release()

    # 브라우저 호환 H.264 mp4로 재인코딩
    _transcode_to_web_h264(temp_path, output_path)
    print(f"분석 영상 저장 완료: {output_path}")
