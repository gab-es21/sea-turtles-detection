from ultralytics import YOLO
from datetime import datetime

DATA_YAML = "../Dataset/sea-turtles-1/data.yaml"
IMG_SIZE = 640
BATCH = 8
EPOCHS = 100

def main():
    run_name = f"yolov9c_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model = YOLO("yolov9c.pt")

    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH,
        epochs=EPOCHS,
        device=0,
        project="runs/train",
        name=run_name,
        workers=0,
    )

if __name__ == "__main__":
    main()
