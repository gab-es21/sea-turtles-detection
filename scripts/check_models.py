from ultralytics import YOLO
from datetime import datetime
import os
import shutil
import traceback

MODELS = [
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt",
    "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt",
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

VALID_LOG = os.path.join(RESULTS_DIR, "valid_models.log")
ERROR_LOG = os.path.join(RESULTS_DIR, "check_models_errors.log")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(VALID_LOG, "w") as vlog, open(ERROR_LOG, "w") as elog:
        vlog.write(f"# Model check {datetime.now()}\n\n")

        for model_name in MODELS:
            try:
                local_model_path = os.path.join(MODELS_DIR, model_name)

                if os.path.exists(local_model_path):
                    vlog.write(model_name + " (cached)\n")
                    print(f"✔ {model_name} already exists")
                    continue

                print(f"Downloading {model_name}...")
                model = YOLO(model_name)  # download only

                src = model.ckpt_path
                if not src or not os.path.exists(src):
                    raise RuntimeError("Downloaded checkpoint not found")

                shutil.copy(src, local_model_path)
                vlog.write(model_name + "\n")
                print(f"✔ Stored at models/{model_name}")

            except Exception:
                print(f"✖ Failed {model_name}")
                elog.write(f"\n[{datetime.now()}] {model_name}\n")
                elog.write(traceback.format_exc())
                elog.write("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
