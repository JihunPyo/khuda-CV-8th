from __future__ import annotations

import sys
from pathlib import Path

from demo_bbox import make_split_bbox_demo
from demo_skeleton import make_person_skeleton_videos_from_npz


DEFAULT_VIDEO = "inputs/normal_test2.mp4"

WARNING_DELAY_SEC = 5.0
WARNING_HOLD_SEC = 5.0  # None이면 끝까지 유지
WARNING_TEXT = "NO DISABLED PERSON"


def resolve_video_path(base_dir: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base_dir / pp).resolve()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    base = Path(__file__).resolve().parent
    in_arg = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_VIDEO
    video_path = resolve_video_path(base, in_arg)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    stem = video_path.stem
    out_root = base / "outputs" / stem

    botsort_dir = out_root / "botsort"
    inference_dir = out_root / "inference"
    person_video_dir = out_root / "person_videos"
    final_demo_dir = out_root / "final_demo"
    crop_ske_dir = out_root / "crop_skeleton_videos"

    ensure_dir(final_demo_dir)
    ensure_dir(crop_ske_dir)

    bboxes_csv = botsort_dir / f"{stem}_bboxes.csv"
    pred_summary = inference_dir / f"{stem}_pred_summary.csv"
    demo1_out = final_demo_dir / f"{stem}_split_bbox.mp4"

    print("\n" + "=" * 60)
    print(">>> [Demo #1] split bbox + delayed warning")
    print("=" * 60)

    make_split_bbox_demo(
        input_video=str(video_path),
        bboxes_csv=str(bboxes_csv),
        pred_summary_csv=str(pred_summary),
        out_path=str(demo1_out),
        warning_text=WARNING_TEXT,
        frame_base=1,
        warning_delay_sec=WARNING_DELAY_SEC,
        warning_hold_sec=WARNING_HOLD_SEC,
        warning_mode="after_last_new_person",  # ✅ 핵심
    )
    print(f"[Saved] {demo1_out}")

    print("\n" + "=" * 60)
    print(">>> [Demo #2] per-person skeleton overlay videos")
    print("=" * 60)

    make_person_skeleton_videos_from_npz(
        person_video_dir=str(person_video_dir),
        inference_npz_dir=str(inference_dir),
        out_dir=str(crop_ske_dir),
        prefix=stem,
        overlay=True,
    )
    print(f"[Saved] {crop_ske_dir}")


if __name__ == "__main__":
    main()
