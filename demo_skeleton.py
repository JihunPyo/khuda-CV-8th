# demo_skeleton.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# 간단한 포즈 연결(33 기준 일부만)
POSE_EDGES = [
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 12),            # shoulders
    (11, 23), (12, 24),  # torso
    (23, 24),            # hips
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_pt(pt: np.ndarray) -> Optional[Tuple[int, int]]:
    """pt = [x_px, y_px]"""
    if pt is None:
        return None
    if np.any(np.isnan(pt)):
        return None
    return int(pt[0]), int(pt[1])


def draw_skeleton(img: np.ndarray, pts_px: np.ndarray) -> None:
    """
    pts_px: (33,2) pixel coords. NaN 포함 가능.
    """
    # 점
    for i in range(min(len(pts_px), 33)):
        p = _safe_pt(pts_px[i])
        if p is None:
            continue
        cv2.circle(img, p, 3, (255, 255, 255), -1)

    # 선
    for a, b in POSE_EDGES:
        if a >= len(pts_px) or b >= len(pts_px):
            continue
        pa = _safe_pt(pts_px[a])
        pb = _safe_pt(pts_px[b])
        if pa is None or pb is None:
            continue
        cv2.line(img, pa, pb, (255, 255, 255), 2)


def _load_landmarks_npz(npz_path: Path) -> Optional[np.ndarray]:
    """
    가능한 키들을 최대한 호환:
      - image_landmarks: (T,33,?>=2)
      - landmarks:       (T,33,?>=2)
    반환: (T,33,2) normalized xy
    """
    if not npz_path.exists():
        return None
    data = np.load(str(npz_path), allow_pickle=True)
    key = None
    for k in ["image_landmarks", "landmarks"]:
        if k in data.files:
            key = k
            break
    if key is None:
        return None

    arr = data[key]
    if arr.ndim != 3 or arr.shape[1] < 33 or arr.shape[2] < 2:
        return None

    xy = arr[:, :33, :2].astype(np.float32)  # (T,33,2)
    return xy


def make_person_skeleton_videos_from_npz(
    person_video_dir: str,
    inference_npz_dir: str,
    out_dir: str,
    prefix: str = "",
    overlay: bool = True,
) -> List[str]:
    """
    person_video_dir: 사람별 mp4들이 있는 폴더 (id_0063.mp4 또는 normal_id_0063.mp4)
    inference_npz_dir: npz들이 있는 폴더 (id_0063.npz 또는 normal_id_0063.npz)
    """
    pdir = Path(person_video_dir)
    idir = Path(inference_npz_dir)
    odir = Path(out_dir)
    ensure_dir(odir)

    outs: List[str] = []
    mp4s = sorted(pdir.glob("*.mp4"))
    for mp4 in mp4s:
        stem = mp4.stem  # id_0001 or normal_id_0001

        # ✅ 매칭을 유연하게:
        # 1) 같은 stem
        # 2) prefix_stem
        npz_candidates = [
            idir / f"{stem}.npz",
            (idir / f"{prefix}_{stem}.npz") if prefix else None,
        ]
        npz_candidates = [x for x in npz_candidates if x is not None]
        npz_path = next((x for x in npz_candidates if x.exists()), None)

        if npz_path is None:
            print(f"[SKIP] npz not found for {mp4.name}")
            continue

        lms_xy = _load_landmarks_npz(npz_path)
        if lms_xy is None:
            print(f"[SKIP] npz has no valid landmarks: {npz_path.name}")
            continue

        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            print(f"[SKIP] cannot open video: {mp4.name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-6:
            fps = 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_name = f"{prefix}_{stem}.mp4" if prefix and not stem.startswith(prefix + "_") else f"{stem}.mp4"
        out_path = odir / out_name
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        t = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if overlay and t < lms_xy.shape[0]:
                pts = lms_xy[t]  # (33,2) normalized
                pts_px = np.empty((33, 2), dtype=np.float32)
                pts_px[:, 0] = pts[:, 0] * W
                pts_px[:, 1] = pts[:, 1] * H
                draw_skeleton(frame, pts_px)

            writer.write(frame)
            t += 1

        cap.release()
        writer.release()
        outs.append(str(out_path))
        print(f"[Saved] {out_path}")

    return outs
