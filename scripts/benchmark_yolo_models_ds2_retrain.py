from ultralytics import YOLO
from datetime import datetime
import time
import csv
import os
import traceback

# ================= PATH SETUP =================
# Phase 6 — Retrain top 3 models from DS2 benchmark with corrected hyperparameters.
# Selected models: yolo26m, yolov9m, yolo26s (best mAP50 × Recall score).
# Key changes vs Phase 5: batch=16, workers=4, patience=50, epochs=300 ceiling.
# Runs and results are kept separate (retrain_ds2/) to preserve Phase 5 data.

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "retrain_ds2")
RUNS_DIR    = os.path.join(BASE_DIR, "runs", "retrain_ds2")

DATA_YAML = os.path.abspath(
    os.path.join(BASE_DIR, "Dataset", "sea-turtles-2", "data.yaml")
)

# ================= TRAIN CONFIG =================
IMG_SIZE = 640
BATCH    = 16    # increased from 8 — more stable gradients on heterogeneous DS2
EPOCHS   = 300   # ceiling — early stopping (patience) will stop before this
PATIENCE = 50    # stop if val mAP50 does not improve for 50 consecutive epochs
DEVICE   = 0
WORKERS  = 4     # parallel data prefetch — fixes GPU underutilisation seen in Phase 5
SEED     = 42

# Top 3 models by mAP50 × Recall from Phase 5 DS2 benchmark
MODELS = [
    "yolo26m.pt",   # mAP50=0.504, Recall=0.800  score=0.403
    "yolov9m.pt",   # mAP50=0.463, Recall=0.700  score=0.324
    "yolo26s.pt",   # mAP50=0.460, Recall=0.667  score=0.307
]

RESULTS_CSV = os.path.join(RESULTS_DIR, "retrain_results_ds2.csv")
ERROR_LOG   = os.path.join(RESULTS_DIR, "retrain_errors_ds2.log")
# ==============================================


def log_error(model_name):
    with open(ERROR_LOG, "a") as f:
        f.write(f"\n[{datetime.now()}] {model_name}\n")
        f.write(traceback.format_exc())
        f.write("\n" + "-" * 80 + "\n")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model",
                "epochs_run",
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

        print(f"\n=== Retraining {model_name} (DS2 retrain) ===")

        if not os.path.exists(model_path):
            print(f"✖ Model not found: {model_path}")
            continue

        try:
            run_name = f"{model_name}_retrain_ds2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            model = YOLO(model_path)

            start = time.time()

            model.train(
                data=DATA_YAML,
                imgsz=IMG_SIZE,
                batch=BATCH,
                epochs=EPOCHS,
                patience=PATIENCE,
                device=DEVICE,
                workers=WORKERS,
                seed=SEED,
                project=RUNS_DIR,
                name=run_name,
            )

            train_time_min = (time.time() - start) / 60

            metrics = model.val()

            # Read actual epochs run from results (early stopping may have stopped early)
            results_csv_path = os.path.join(RUNS_DIR, run_name, "results.csv")
            epochs_run = EPOCHS
            if os.path.exists(results_csv_path):
                with open(results_csv_path) as rf:
                    epochs_run = sum(1 for line in rf) - 1  # subtract header

            with open(RESULTS_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    model_name,
                    epochs_run,
                    round(train_time_min, 2),
                    metrics.box.map50,
                    metrics.box.map,
                    metrics.box.mp,
                    metrics.box.mr,
                    run_name
                ])

            print(f"✔ Finished {model_name} in {train_time_min:.2f} min ({epochs_run} epochs)")

        except Exception:
            print(f"✖ Failed {model_name}, skipping...")
            log_error(model_name)
            continue


if __name__ == "__main__":
    main()
