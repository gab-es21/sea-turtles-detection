from ultralytics import YOLO
import csv
import os
from pathlib import Path

# ================= PATH SETUP =================
BASE_DIR = Path(__file__).resolve().parent.parent

RUNS_DIR   = BASE_DIR / "runs" / "train"
RESULTS_DIR = BASE_DIR / "results"
DATA_YAML  = BASE_DIR.parent / "Dataset" / "sea-turtles-1" / "data.yaml"

RESULTS_CSV = RESULTS_DIR / "test_results.csv"

# Same order as benchmark
MODELS = [
    "yolo26n", "yolo26s", "yolo26m",
    "yolo11n", "yolo11s", "yolo11m",
    "yolov10n", "yolov10s", "yolov10m",
    "yolov9t", "yolov9s", "yolov9m", "yolov9c",
    "yolov8n", "yolov8s", "yolov8m",
]

# Map model name → run folder (from benchmark run names)
RUN_NAMES = {
    "yolo26n":  "yolo26n_20260202_011017",
    "yolo26s":  "yolo26s_20260202_022129",
    "yolo26m":  "yolo26m_20260202_033138",
    "yolo11n":  "yolo11n_20260202_050518",
    "yolo11s":  "yolo11s_20260202_055950",
    "yolo11m":  "yolo11m_20260202_065642",
    "yolov10n": "yolov10n_20260202_081804",
    "yolov10s": "yolov10s_20260202_092131",
    "yolov10m": "yolov10m_20260202_102932",
    "yolov9t":  "yolov9t_20260202_120044",
    "yolov9s":  "yolov9s_20260202_132754",
    "yolov9m":  "yolov9m_20260202_145822",
    "yolov9c":  "yolov9c_20260202_163426",
    "yolov8n":  "yolov8n_20260202_182141",
    "yolov8s":  "yolov8s_20260202_191241",
    "yolov8m":  "yolov8m_20260202_200749",
}
# ==============================================


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "split",
            "mAP50",
            "mAP50_95",
            "precision",
            "recall",
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
        ])

    for model_name in MODELS:
        run_name = RUN_NAMES[model_name]
        weights  = RUNS_DIR / run_name / "weights" / "best.pt"

        if not weights.exists():
            print(f"[SKIP] {model_name} — weights not found: {weights}")
            continue

        print(f"\n=== Evaluating {model_name} on test split ===")
        model = YOLO(str(weights))

        metrics = model.val(
            data=str(DATA_YAML),
            split="test",
            imgsz=640,
            batch=8,
            device=0,
            workers=0,
            verbose=True,
            plots=True,
            project=str(RESULTS_DIR / "test_runs"),
            name=model_name,
        )

        speed = metrics.speed  # dict: preprocess / inference / postprocess (ms/img)

        row = [
            model_name,
            "test",
            round(metrics.box.map50, 6),
            round(metrics.box.map,   6),
            round(metrics.box.mp,    6),
            round(metrics.box.mr,    6),
            round(speed.get("preprocess",  0), 3),
            round(speed.get("inference",   0), 3),
            round(speed.get("postprocess", 0), 3),
        ]

        with open(RESULTS_CSV, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(f"  mAP50={row[2]:.4f}  mAP50-95={row[3]:.4f}  "
              f"P={row[4]:.4f}  R={row[5]:.4f}  "
              f"inference={row[7]:.2f}ms/img")

    print(f"\nDone. Results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
