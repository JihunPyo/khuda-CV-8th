# demo_bbox.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


# -------------------------
# IO
# -------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_bboxes_csv(path: str, frame_base: int = 1) -> Dict[int, List[Tuple[int, int, int, int, int]]]:
    """
    returns:
      bbox_map[frame] = [(stable_id, x1,y1,x2,y2), ...]
    """
    bbox_map: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"bboxes_csv not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # tracker가 frame을 1부터 썼으면 frame_base=1로 맞춤
            fr = int(row["frame"]) - (frame_base - 1)
            sid = int(row.get("stable_id") or row.get("raw_id") or 0)
            x = int(float(row["x"]))
            y = int(float(row["y"]))
            w = int(float(row["w"]))
            h = int(float(row["h"]))
            x1, y1, x2, y2 = x, y, x + w, y + h
            bbox_map.setdefault(fr, []).append((sid, x1, y1, x2, y2))
    return bbox_map


def load_pred_summary(path: str) -> Dict[int, int]:
    """
    pred_map[stable_id] = pred_class (0=normal, 1=abnormal)
    """
    pred_map: Dict[int, int] = {}
    p = Path(path)
    if not p.exists():
        # pred_summary가 없으면 전부 normal로 처리되는 효과
        return pred_map

    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = int(row["stable_id"])
            pred = int(row["pred_class"])
            pred_map[sid] = pred
    return pred_map


# -------------------------
# Draw
# -------------------------
def _draw_bbox(img: np.ndarray, sid: int, xyxy: Tuple[int, int, int, int], is_disabled: bool) -> None:
    x1, y1, x2, y2 = xyxy
    color = (0, 0, 255) if is_disabled else (0, 255, 0)  # red vs green
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        f"id_{sid:04d}",
        (x1, max(0, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_warning(img: np.ndarray, text: str) -> None:
    # 디자인은 기존 스타일 유지: 빨간 글씨, 상단 중앙
    H, W = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4
    thickness = 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, (W - tw) // 2)
    y = 60
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)


# -------------------------
# Main
# -------------------------
def make_split_bbox_demo(
    input_video: str,
    bboxes_csv: str,
    pred_summary_csv: str,
    out_path: str,
    warning_text: str = "NO DISABLED PERSON",
    frame_base: int = 1,
    # ✅ 추가: 경고 타이밍
    warning_delay_sec: float = 0.0,
    warning_hold_sec: Optional[float] = None,
    # ✅ 모드:
    #   - "frame": (기존) 해당 프레임에 disabled 박스가 없으면 즉시 warning
    #   - "after_last_new_person": (추천) "새로 등장한 사람"이 마지막으로 나온 뒤 delay_sec 지나면 warning
    warning_mode: str = "frame",
) -> str:
    bbox_map = load_bboxes_csv(bboxes_csv, frame_base=frame_base)
    pred_map = load_pred_summary(pred_summary_csv)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 30.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_p = Path(out_path)
    _ensure_dir(out_p.parent)

    writer = cv2.VideoWriter(
        str(out_p),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W * 2, H),
    )

    # -------------------------
    # ✅ "after_last_new_person"용: 마지막 사람 등장 프레임 계산
    # -------------------------
    all_sids = set()
    first_frame_of_sid: Dict[int, int] = {}
    for fr, items in bbox_map.items():
        for sid, *_ in items:
            all_sids.add(sid)
            if sid not in first_frame_of_sid:
                first_frame_of_sid[sid] = fr

    last_new_person_frame = max(first_frame_of_sid.values()) if first_frame_of_sid else 1
    delay_frames = int(round(warning_delay_sec * fps))
    hold_frames = int(round(warning_hold_sec * fps)) if warning_hold_sec is not None else None

    warn_start = last_new_person_frame + delay_frames
    warn_end = (warn_start + hold_frames) if hold_frames is not None else None

    # ✅ "장애인(Abnormal) 존재 여부"를 영상 전체 기준으로 잡아두기
    any_disabled_global = any(pred_map.get(sid, 0) == 1 for sid in all_sids)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_idx += 1

        left = frame
        right = frame.copy()

        # bbox draw
        frame_has_disabled = False
        for sid, x1, y1, x2, y2 in bbox_map.get(frame_idx, []):
            pred = pred_map.get(sid, 0)  # 없으면 normal
            is_disabled = (pred == 1)
            frame_has_disabled |= is_disabled
            _draw_bbox(right, sid, (x1, y1, x2, y2), is_disabled)

        # warning draw
        show_warning = False
        if warning_mode == "frame":
            # (기존) 이 프레임에 disabled가 없으면 경고
            show_warning = (not frame_has_disabled)

        elif warning_mode == "after_last_new_person":
            # (추천) "전체 결과가 disabled 없음"일 때만,
            # 마지막 사람 등장 이후 delay가 지나고, hold 범위 안이면 경고
            if not any_disabled_global:
                if frame_idx >= warn_start and (warn_end is None or frame_idx < warn_end):
                    show_warning = True

        else:
            raise ValueError(f"unknown warning_mode: {warning_mode}")

        if show_warning:
            _draw_warning(right, warning_text)

        out_frame = np.hstack([left, right])
        writer.write(out_frame)

    cap.release()
    writer.release()
    return str(out_p)
