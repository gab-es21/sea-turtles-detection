import subprocess
import sys
import os
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
YOLO_SRC = os.path.join(PROJECT_ROOT, "src")
DATA_YAML = os.path.join(PROJECT_ROOT, "data_sea_turtles.yaml")

IMG_SIZE = 640
BATCH = 8
EPOCHS = 50
DEVICE = "0"
WORKERS = "0"          # CRÍTICO no Windows
CACHE = "False"

MODEL_WEIGHTS = "yolov9c.pt"
MODEL_CFG = "models/detect/yolov9-c.yaml"
HYP = "data/hyps/hyp.scratch-high.yaml"

RUN_NAME = f"yolov9c_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "train.log")

# -----------------------------
# COMMAND
# -----------------------------
cmd = [
    sys.executable,
    "train.py",
    "--img", str(IMG_SIZE),
    "--batch", str(BATCH),
    "--epochs", str(EPOCHS),
    "--data", DATA_YAML,
    "--weights", MODEL_WEIGHTS,
    "--cfg", MODEL_CFG,
    "--hyp", HYP,
    "--device", DEVICE,
    "--workers", WORKERS,
    "--cache", CACHE,
    "--project", "runs/train",
    "--name", RUN_NAME,
]

# -----------------------------
# RUN
# -----------------------------
print("Starting YOLOv9 training")
print("Run name:", RUN_NAME)
print("Logging to:", LOG_FILE)
print("Command:\n", " ".join(cmd))

with open(LOG_FILE, "w", encoding="utf-8") as log:
    process = subprocess.Popen(
        cmd,
        cwd=YOLO_SRC,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="")
        log.write(line)

    process.wait()

print("\nTraining finished with return code:", process.returncode)
