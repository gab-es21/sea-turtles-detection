from ultralytics import YOLO

DATA_YAML = "../Dataset/sea-turtles-1/data.yaml"
WEIGHTS = "./runs/detect/runs/train/yolov9c_20260119_122609/weights/best.pt"

def main():
    model = YOLO(WEIGHTS)

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        device=0,
        workers=0,
        batch=8
    )

    print(metrics)

if __name__ == "__main__":
    main()

