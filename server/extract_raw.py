"""
1단계: 탐지 원본값 추출 (GPU 권장)

영상에서 포즈 탐지 "원본값"만 뽑아 작은 .npz 파일로 저장한다.
보정·분석은 이 파일만 있으면 YOLO 없이 다시 돌릴 수 있다.

단독 실행:  python extract_raw.py 영상.mp4 [--pose-model 경로] [--hold-model 경로]
서버에서는 extract() 함수를 직접 호출한다.
"""
import os
import argparse
import numpy as np
import cv2

MAX_PEOPLE = 5          # 프레임당 저장할 최대 인원 (클라이머 선택은 다음 단계에서)
HOLD_SAMPLE_FRAMES = 8  # 홀드 탐지에 쓸 프레임 수

_MODEL_CACHE = {}


def _model(path):
    from ultralytics import YOLO
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = YOLO(path)
    return _MODEL_CACHE[path]


def _device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def tiled_holds(model, img, conf=0.1, overlap=0.15):
    """4분할 + 15% 중첩 + NMS. [x1,y1,x2,y2,score] 반환."""
    h, w = img.shape[:2]
    mw, mh = w // 2, h // 2
    tiles = [
        (0, 0, int(mw * (1 + overlap)), int(mh * (1 + overlap))),
        (int(mw * (1 - overlap)), 0, w, int(mh * (1 + overlap))),
        (0, int(mh * (1 - overlap)), int(mw * (1 + overlap)), h),
        (int(mw * (1 - overlap)), int(mh * (1 - overlap)), w, h),
    ]
    boxes = []
    for x1, y1, x2, y2 in tiles:
        r = model(img[y1:y2, x1:x2], conf=conf, verbose=False)[0]
        for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            boxes.append([b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1, s])
    if not boxes:
        return np.zeros((0, 5), np.float32)
    boxes = np.array(boxes, np.float32)
    keep = cv2.dnn.NMSBoxes(boxes[:, :4].tolist(), boxes[:, 4].tolist(), conf, 0.4)
    return boxes[np.array(keep).flatten()] if len(keep) else np.zeros((0, 5), np.float32)


def extract(video, out_path, pose_model, hold_model=None, pose_conf=0.25, progress=None):
    """영상 → .raw.npz. progress(done, total) 콜백으로 진행률을 알린다."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    device = _device()
    pose = _model(pose_model)

    kpts = np.zeros((n, MAX_PEOPLE, 17, 3), np.float32)
    boxes = np.zeros((n, MAX_PEOPLE, 5), np.float32)
    count = np.zeros(n, np.int16)
    hold_idx = set(np.linspace(0, max(n - 1, 0), HOLD_SAMPLE_FRAMES).astype(int).tolist()) if hold_model else set()
    hold_frames = {}

    i = 0
    while i < n:
        ok, frame = cap.read()
        if not ok:
            break
        if i in hold_idx:
            hold_frames[i] = frame.copy()
        r = pose(frame, conf=pose_conf, device=device, verbose=False)[0]
        if r.keypoints is not None and len(r.boxes) > 0:
            k = r.keypoints.data.cpu().numpy()
            b = np.concatenate([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]], 1)
            order = np.argsort(-b[:, 4])[:MAX_PEOPLE]
            m = len(order)
            kpts[i, :m] = k[order]
            boxes[i, :m] = b[order]
            count[i] = m
        i += 1
        if progress and (i % 10 == 0 or i == n):
            progress(i, n)
    cap.release()
    n = i

    holds, src = [np.zeros((0, 5), np.float32)], [np.zeros(0, int)]
    if hold_model and os.path.exists(hold_model):
        hm = _model(hold_model)
        holds, src = [], []
        for fi in sorted(hold_frames):
            hb = tiled_holds(hm, hold_frames[fi])
            holds.append(hb)
            src.append(np.full(len(hb), fi))

    np.savez_compressed(out_path, fps=fps, size=np.array([w, h]),
                        kpts=kpts[:n], boxes=boxes[:n], count=count[:n],
                        holds=np.concatenate(holds), hold_src_frame=np.concatenate(src))
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--pose-model", default="yolo11l-pose.pt")
    ap.add_argument("--hold-model", default=None)
    a = ap.parse_args()
    out = os.path.splitext(a.video)[0] + ".raw.npz"
    extract(a.video, out, a.pose_model, a.hold_model,
            progress=lambda d, t: print(f"\rpose {d}/{t}", end=""))
    print(f"\n저장 완료: {out}")
