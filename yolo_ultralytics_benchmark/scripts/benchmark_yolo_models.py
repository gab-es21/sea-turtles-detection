from ultralytics import YOLO
from datetime import datetime
import time
import csv
import os
import traceback

# ================= PATH SETUP =================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RUNS_DIR = os.path.join(BASE_DIR, "runs", "train")

DATA_YAML = os.path.abspath(
    os.path.join(BASE_DIR, "..", "Dataset", "sea-turtles-1", "data.yaml")
)

# ================= TRAIN CONFIG =================
IMG_SIZE = 640
BATCH = 8
EPOCHS = 100
DEVICE = 0
WORKERS = 0
SEED = 42

MODELS = [
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt",
    "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt",
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
]

RESULTS_CSV = os.path.join(RESULTS_DIR, "benchmark_results.csv")
ERROR_LOG = os.path.join(RESULTS_DIR, "benchmark_errors.log")
# ==============================================


def log_error(model_name):
    with open(ERROR_LOG, "a") as f:
        f.write(f"\n[{datetime.now()}] {model_name}\n")
        f.write(traceback.format_exc())
        f.write("\n" + "-" * 80 + "\n")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    # CSV header
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model",
                "epochs",
                "train_time_min",
                "mAP50",
                "mAP50_95",
                "precision",
                "recall",
                "run_name"
            ])

    for model_file in MODELS:
        model_path = os.path.join(MODELS_DIR, model_file)
        model_name = model_file.replace(".pt", "")

        print(f"\n=== Processing {model_name} ===")

        if not os.path.exists(model_path):
            print(f"✖ Model not found: {model_path}")
            continue

        try:
            run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 🚫 NO downloads, 🚫 NO ambiguity
            model = YOLO(model_path)

            start = time.time()

            model.train(
                data=DATA_YAML,
                imgsz=IMG_SIZE,
                batch=BATCH,
                epochs=EPOCHS,
                device=DEVICE,
                workers=WORKERS,
                seed=SEED,
                project=RUNS_DIR,
                name=run_name,
            )

            train_time_min = (time.time() - start) / 60

            metrics = model.val()

            with open(RESULTS_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    model_name,
                    EPOCHS,
                    round(train_time_min, 2),
                    metrics.box.map50,
                    metrics.box.map,
                    metrics.box.mp,
                    metrics.box.mr,
                    run_name
                ])

            print(f"✔ Finished {model_name} in {train_time_min:.2f} min")

        except Exception:
            print(f"✖ Failed {model_name}, skipping...")
            log_error(model_name)
            continue


if __name__ == "__main__":
    main()
