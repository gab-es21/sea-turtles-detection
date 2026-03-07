"""
Phase 4 — ByteTrack tracking with YOLOv9m on sea turtle footage.

Usage:
    # Run on a video file:
    python tracking.py --source path/to/video.mp4

    # Run on a folder of images (used as a pseudo-sequence, sorted by name):
    python tracking.py --source path/to/image_folder --fps 30

    # Override default model or tracker config:
    python tracking.py --source video.mp4 --weights path/to/best.pt --tracker bytetrack_turtles.yaml

Outputs (saved under yolo_ultralytics_benchmark/results/tracking/<run_name>/):
    - annotated video (or frames) with track IDs overlaid
    - tracking_stats.txt  with FPS, unique IDs, per-frame ID counts
"""

import argparse
import time
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS  = BASE_DIR / "runs" / "train" / "yolov9m_20260202_145822" / "weights" / "best.pt"
TRACKER  = Path(__file__).resolve().parent / "bytetrack_turtles.yaml"
OUT_DIR  = BASE_DIR / "results" / "tracking"


def parse_args():
    parser = argparse.ArgumentParser(description="ByteTrack sea turtle tracker")
    parser.add_argument("--source",  required=True,
                        help="Video file or folder of images")
    parser.add_argument("--weights", default=str(WEIGHTS),
                        help="Path to best.pt (default: yolov9m)")
    parser.add_argument("--tracker", default=str(TRACKER),
                        help="ByteTrack YAML config")
    parser.add_argument("--conf",    type=float, default=0.30,
                        help="Detection confidence threshold (default: 0.30)")
    parser.add_argument("--iou",     type=float, default=0.50,
                        help="NMS IoU threshold (default: 0.50)")
    parser.add_argument("--imgsz",   type=int,   default=640,
                        help="Inference image size (default: 640)")
    parser.add_argument("--fps",     type=float, default=30.0,
                        help="FPS to stamp on image-folder pseudo-sequences (default: 30)")
    parser.add_argument("--device",  default="0",
                        help="Device: 0 for GPU, cpu for CPU (default: 0)")
    parser.add_argument("--name",    default="run",
                        help="Output sub-folder name (default: run)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── lazy import so help text works without ultralytics installed ──
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not found. Activate the project venv first.")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        sys.exit(f"Weights not found: {weights_path}")

    tracker_path = Path(args.tracker)
    if not tracker_path.exists():
        sys.exit(f"Tracker config not found: {tracker_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    project_dir = OUT_DIR
    run_name    = args.name

    print(f"Model   : {weights_path}")
    print(f"Tracker : {tracker_path}")
    print(f"Source  : {args.source}")
    print(f"Output  : {project_dir / run_name}")
    print()

    model = YOLO(str(weights_path))

    # ── run tracking ──────────────────────────────────────────────────
    results_gen = model.track(
        source=args.source,
        tracker=str(tracker_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        save=True,
        project=str(project_dir),
        name=run_name,
        verbose=False,
    )

    frame_count  = 0
    all_ids      = set()
    id_switches  = 0          # frames where track count changes (proxy for activity)
    prev_ids     = set()
    per_frame    = []         # (frame_idx, set_of_ids)

    t_start = time.time()

    for result in results_gen:
        frame_count += 1
        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            frame_ids = set(int(i) for i in boxes.id.tolist())
        else:
            frame_ids = set()

        all_ids.update(frame_ids)
        if frame_ids != prev_ids:
            id_switches += 1
        prev_ids = frame_ids
        per_frame.append((frame_count, frame_ids))

    elapsed = time.time() - t_start
    fps     = frame_count / elapsed if elapsed > 0 else float("nan")

    # ── report ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Frames processed : {frame_count}")
    print(f"Elapsed time     : {elapsed:.2f} s")
    print(f"End-to-end FPS   : {fps:.1f}")
    print(f"Unique track IDs : {len(all_ids)}")
    print(f"ID set changes   : {id_switches}  (proxy — not formal ID-switch metric)")
    print(f"{'='*50}\n")

    # ── save stats ───────────────────────────────────────────────────
    stats_path = project_dir / run_name / "tracking_stats.txt"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        f.write(f"model         : {weights_path.name}\n")
        f.write(f"tracker       : {tracker_path.name}\n")
        f.write(f"source        : {args.source}\n")
        f.write(f"conf          : {args.conf}\n")
        f.write(f"iou           : {args.iou}\n")
        f.write(f"frames        : {frame_count}\n")
        f.write(f"elapsed_s     : {elapsed:.3f}\n")
        f.write(f"fps           : {fps:.2f}\n")
        f.write(f"unique_ids    : {len(all_ids)}\n")
        f.write(f"id_set_changes: {id_switches}\n")
        f.write("\nper_frame_ids:\n")
        for fno, ids in per_frame:
            f.write(f"  frame {fno:4d}: {sorted(ids)}\n")

    print(f"Stats saved to: {stats_path}")
    print(f"Annotated output in: {project_dir / run_name}/")


if __name__ == "__main__":
    main()
