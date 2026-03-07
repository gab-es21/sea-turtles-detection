# Sea Turtles Detection

Detect sea turtles from drone imagery using YOLO object detection models.

## Overview

This project trains and benchmarks multiple YOLO architectures for detecting sea turtles in aerial/drone images. It uses a dataset from Roboflow containing annotated NIR (near-infrared) drone footage of sea turtles on nesting beaches.

## Dataset

Two dataset variants were used across experiments:

| ID | Name      | Train  | Val   | Test  | Notes                |
| -- | --------- | ------ | ----- | ----- | -------------------- |
| 1  | Base (B)  | 968    | 272   | 118   | Original annotations |
| 2  | Tiled (W) | 15,488 | 4,352 | 1,888 | Tiled augmentation   |

- **Class**: `Turtle` (single-class detection)
- **Source**: [Roboflow — sea-turtles-yia2e](https://universe.roboflow.com/gabriel-esteves-dy2cw/sea-turtles-yia2e/dataset/1)
- **License**: CC BY 4.0

The dataset is stored locally under `Dataset/sea-turtles-1/` and is excluded from version control (see `.gitignore`).

## Benchmark Results

All models trained for **100 epochs** on Dataset 1 (Base), image size 640, batch 8, seed 42.

| Model    | mAP50     | mAP50-95 | Precision | Recall | Train Time (min) |
| -------- | --------- | -------- | --------- | ------ | ---------------- |
| yolov9c  | **0.955** | 0.633    | 0.935     | 0.910  | 106.8            |
| yolo26m  | 0.952     | 0.643    | 0.927     | 0.897  | 93.2             |
| yolov9m  | 0.950     | 0.619    | 0.913     | 0.900  | 95.4             |
| yolov11m | 0.946     | 0.616    | 0.896     | 0.915  | 81.0             |
| yolov8m  | 0.946     | 0.613    | 0.922     | 0.910  | 78.2             |
| yolov10m | 0.944     | 0.619    | 0.924     | 0.903  | 90.8             |
| yolo26s  | 0.940     | 0.593    | 0.860     | 0.913  | 69.6             |
| yolov10s | 0.936     | 0.589    | 0.892     | 0.893  | 67.6             |
| yolov8s  | 0.936     | 0.567    | 0.905     | 0.866  | 54.7             |
| yolov11s | 0.932     | 0.574    | 0.892     | 0.882  | 56.5             |
| yolov9s  | 0.925     | 0.559    | 0.875     | 0.856  | 89.9             |
| yolov11n | 0.907     | 0.524    | 0.857     | 0.838  | 54.2             |
| yolov10n | 0.907     | 0.525    | 0.845     | 0.854  | 63.1             |
| yolo26n  | 0.887     | 0.508    | 0.832     | 0.831  | 70.8             |
| yolov9t  | 0.886     | 0.497    | 0.840     | 0.808  | 86.8             |
| yolov8n  | 0.886     | 0.500    | 0.854     | 0.794  | 50.6             |

Full results: [`yolo_ultralytics_benchmark/results/benchmark_results.csv`](yolo_ultralytics_benchmark/results/benchmark_results.csv)

## Earlier Experiments

| Run | Architecture | Epochs | Precision | Recall | mAP50 | Dataset | Notes |
| --- | ------------ | ------ | --------- | ------ | ----- | ------- | ----- |
| 1   | YOLOv8n      | 50     | 0.748     | 0.764  | 0.809 | 1       | Colab |
| 2   | YOLOv8n      | 5      | 0.652     | 0.419  | 0.453 | 2       | Colab |
| 3   | YOLOv8n      | 150    | 0.830     | 0.920  | 0.825 | 1       | Local |
| 4   | YOLOv8n      | 400    | 0.824     | 0.920  | 0.823 | 1       | Local |
| 5   | YOLOv8n      | 100    | 0.786     | 0.910  | 0.827 | 2       | Local |
| 7   | YOLOv8m      | 100    | 0.866     | 0.666  | 0.796 | 1       | Local |

## Project Structure

```text
sea-turtles-detection/
├── Dataset/
│   └── sea-turtles-1/          # YOLO-format dataset (gitignored)
│       ├── train/images/
│       ├── valid/images/
│       ├── test/images/
│       └── data.yaml
│
├── yolo_ultralytics_benchmark/ # Multi-model benchmark (v8/v9/v10/v11/v26)
│   ├── models/                 # Pre-trained .pt weights (gitignored)
│   ├── scripts/
│   │   ├── benchmark_yolo_models.py  # Main benchmark runner
│   │   └── check_models.py
│   ├── results/
│   │   └── benchmark_results.csv
│   └── runs/                   # Training outputs (gitignored)
│
├── archive/                    # Previous experiments
│   ├── yolov8/                 # Colab notebooks (train/val/detect)
│   ├── yolov8_local/           # Local notebooks + hyperparameter tuning
│   ├── yolov8_tile/            # Tiled dataset experiments (Colab)
│   ├── yolov8_tile_local/      # Tiled dataset experiments (local)
│   ├── yolov5_local/           # YOLOv5 baseline
│   ├── yolov9/                 # YOLOv9 training notebooks
│   ├── yolov9_ultralytics/     # YOLOv9 via Ultralytics API
│   └── yolov26n_ultralytics/   # YOLO26n via Ultralytics API
│
├── runs/                       # Detection inference outputs (gitignored)
├── env.sample                  # Environment variable template
└── Makefile                    # venv activation helpers
```

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Install dependencies

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate     # Linux/macOS
pip install ultralytics
```

### Environment variables

Copy `env.sample` to `.env` and fill in your API keys:

```bash
cp env.sample .env
```

Required keys:

- `ROBOFLOW_API_KEY` — for dataset access
- `COMET_API_KEY` — for experiment tracking (optional)

## Running the Benchmark

```bash
cd yolo_ultralytics_benchmark
python scripts/benchmark_yolo_models.py
```

Results are appended to `yolo_ultralytics_benchmark/results/benchmark_results.csv`.

Pre-trained model weights must be placed in `yolo_ultralytics_benchmark/models/` before running.

## Running Inference

```bash
yolo detect predict model=yolo_ultralytics_benchmark/models/yolov9c.pt source=<image_or_folder>
```

## Training Results Analysis

All 16 models were trained for 100 epochs on the same dataset and hardware. Results are stored in [`yolo_ultralytics_benchmark/results/images/`](yolo_ultralytics_benchmark/results/images/).

### Summary Table

| Model    | mAP50     | mAP50-95 | Precision | Recall | Train Time (min) | Efficiency (mAP50/min) |
| -------- | --------- | -------- | --------- | ------ | ---------------- | ---------------------- |
| yolov9c  | **0.955** | 0.633    | 0.935     | 0.910  | 106.8            | 0.00894                |
| yolo26m  | 0.952     | **0.643** | 0.927     | 0.897  | 93.2             | 0.01022                |
| yolov9m  | 0.950     | 0.619    | 0.913     | 0.900  | 95.4             | 0.00996                |
| yolov11m | 0.946     | 0.616    | 0.896     | **0.915** | 81.0           | 0.01168                |
| yolov8m  | 0.946     | 0.613    | 0.922     | 0.910  | 78.2             | 0.01210                |
| yolov10m | 0.944     | 0.619    | **0.924** | 0.903  | 90.8             | 0.01040                |
| yolo26s  | 0.940     | 0.593    | 0.860     | 0.913  | 69.6             | 0.01351                |
| yolov10s | 0.936     | 0.589    | 0.892     | 0.893  | 67.6             | 0.01385                |
| yolov8s  | 0.936     | 0.567    | 0.905     | 0.866  | 54.7             | 0.01711                |
| yolov11s | 0.932     | 0.574    | 0.892     | 0.882  | 56.5             | 0.01650                |
| yolov9s  | 0.925     | 0.559    | 0.875     | 0.856  | 89.9             | 0.01029                |
| yolov11n | 0.907     | 0.524    | 0.857     | 0.838  | 54.2             | 0.01674                |
| yolov10n | 0.907     | 0.525    | 0.845     | 0.854  | 63.1             | 0.01437                |
| yolo26n  | 0.887     | 0.508    | 0.832     | 0.831  | 70.8             | 0.01253                |
| yolov9t  | 0.886     | 0.497    | 0.840     | 0.808  | 86.8             | 0.01021                |
| yolov8n  | 0.886     | 0.500    | 0.854     | 0.794  | **50.6**         | 0.01751                |

> **Efficiency** = mAP50 / train time. Higher is better for resource-constrained deployments.

### Key Observations

- **Best accuracy**: YOLOv9c achieves the highest mAP50 (0.955) and strong precision (0.935), at the cost of the longest training time (106.8 min).
- **Best mAP50-95**: YOLO26m leads at 0.643, meaning it localises turtles more precisely (tighter bounding boxes).
- **Best efficiency**: YOLOv8n delivers the fastest training (50.6 min) and competitive mAP50 (0.886) — best for rapid iteration.
- **Best small model**: YOLOv8s reaches 0.936 mAP50 in only 54.7 min, the most efficient small model overall.
- **YOLOv9 family**: Consistently strong across all sizes; the gap between `t → s → m → c` is large (~0.040 mAP50), making model size selection important.
- **YOLO11 vs YOLOv8**: YOLO11 trains in similar time but achieves higher recall across all sizes, suggesting better detection coverage with fewer misses.
- **YOLO26**: The newest architecture; the medium variant rivals YOLOv9c in mAP50-95 while training ~13 min faster.
- **Nano models** (v8n, v9t, v10n, v11n, v26n): All cluster around 0.886–0.907 mAP50. YOLOv11n and YOLOv10n lead the nano tier.

---

### Per-Model Results

#### YOLOv9c — Best Overall (mAP50: 0.955)

| Metric     | Value |
| --- | --- |
| mAP50      | 0.955 |
| mAP50-95   | 0.633 |
| Precision  | 0.935 |
| Recall     | 0.910 |
| Train time | 106.8 min |

Training curves show steady convergence with no overfitting. Loss continues decreasing through epoch 100.

![YOLOv9c results](yolo_ultralytics_benchmark/results/images/yolov9c/results.png)
![YOLOv9c confusion matrix](yolo_ultralytics_benchmark/results/images/yolov9c/confusion_matrix_normalized.png)
![YOLOv9c PR curve](yolo_ultralytics_benchmark/results/images/yolov9c/BoxPR_curve.png)
![YOLOv9c validation predictions](yolo_ultralytics_benchmark/results/images/yolov9c/val_batch0_pred.jpg)

---

#### YOLO26m — Best mAP50-95 (0.643)

| Metric     | Value |
| --- | --- |
| mAP50      | 0.952 |
| mAP50-95   | 0.643 |
| Precision  | 0.927 |
| Recall     | 0.897 |
| Train time | 93.2 min |

Highest mAP50-95 of all models, indicating the tightest bounding box localisation. Trains ~13 min faster than YOLOv9c for comparable mAP50.

![YOLO26m results](yolo_ultralytics_benchmark/results/images/yolo26m/results.png)
![YOLO26m confusion matrix](yolo_ultralytics_benchmark/results/images/yolo26m/confusion_matrix_normalized.png)
![YOLO26m validation predictions](yolo_ultralytics_benchmark/results/images/yolo26m/val_batch0_pred.jpg)

---

#### YOLOv9m

| Metric     | Value |
| --- | --- |
| mAP50      | 0.950 |
| mAP50-95   | 0.619 |
| Precision  | 0.913 |
| Recall     | 0.900 |
| Train time | 95.4 min |

Very close to YOLOv9c in accuracy (~0.005 mAP50 gap) but with a smaller model. A strong choice when model size matters.

![YOLOv9m results](yolo_ultralytics_benchmark/results/images/yolov9m/results.png)
![YOLOv9m validation predictions](yolo_ultralytics_benchmark/results/images/yolov9m/val_batch0_pred.jpg)

---

#### YOLO11m

| Metric     | Value |
| --- | --- |
| mAP50      | 0.946 |
| mAP50-95   | 0.616 |
| Precision  | 0.896 |
| Recall     | 0.915 |
| Train time | 81.0 min |

Best recall among medium models (0.915), meaning fewest missed turtles. Trains significantly faster than YOLOv9c/m for only ~0.009 mAP50 loss.

![YOLO11m results](yolo_ultralytics_benchmark/results/images/yolo11m/results.png)
![YOLO11m validation predictions](yolo_ultralytics_benchmark/results/images/yolo11m/val_batch0_pred.jpg)

---

#### YOLOv8m

| Metric     | Value |
| --- | --- |
| mAP50      | 0.946 |
| mAP50-95   | 0.613 |
| Precision  | 0.922 |
| Recall     | 0.910 |
| Train time | 78.2 min |

Fastest of the medium models. Matches YOLO11m in mAP50 while training ~3 min faster, with higher precision but slightly lower recall.

![YOLOv8m results](yolo_ultralytics_benchmark/results/images/yolov8m/results.png)
![YOLOv8m validation predictions](yolo_ultralytics_benchmark/results/images/yolov8m/val_batch0_pred.jpg)

---

#### YOLOv10m

| Metric     | Value |
| --- | --- |
| mAP50      | 0.944 |
| mAP50-95   | 0.619 |
| Precision  | 0.924 |
| Recall     | 0.903 |
| Train time | 90.8 min |

Highest precision among medium models (0.924). Good mAP50-95 matching YOLOv9m, but takes longer to train than v8m/v11m for similar mAP50.

![YOLOv10m results](yolo_ultralytics_benchmark/results/images/yolov10m/results.png)
![YOLOv10m validation predictions](yolo_ultralytics_benchmark/results/images/yolov10m/val_batch0_pred.jpg)

---

#### YOLO26s

| Metric     | Value |
| --- | --- |
| mAP50      | 0.940 |
| mAP50-95   | 0.593 |
| Precision  | 0.860 |
| Recall     | 0.913 |
| Train time | 69.6 min |

Impressive for a small model — nearly matches YOLOv10m mAP50 in 69.6 min. High recall (0.913) makes it a strong candidate when minimising missed detections.

![YOLO26s results](yolo_ultralytics_benchmark/results/images/yolo26s/results.png)
![YOLO26s validation predictions](yolo_ultralytics_benchmark/results/images/yolo26s/val_batch0_pred.jpg)

---

#### YOLOv10s

| Metric     | Value |
| --- | --- |
| mAP50      | 0.936 |
| mAP50-95   | 0.589 |
| Precision  | 0.892 |
| Recall     | 0.893 |
| Train time | 67.6 min |

Balanced small model. Well-rounded precision/recall with fast training.

![YOLOv10s results](yolo_ultralytics_benchmark/results/images/yolov10s/results.png)
![YOLOv10s validation predictions](yolo_ultralytics_benchmark/results/images/yolov10s/val_batch0_pred.jpg)

---

#### YOLOv8s — Best Small Model Efficiency

| Metric     | Value |
| --- | --- |
| mAP50      | 0.936 |
| mAP50-95   | 0.567 |
| Precision  | 0.905 |
| Recall     | 0.866 |
| Train time | 54.7 min |

Highest mAP50 per minute among small models. Excellent choice when training time and accuracy must both be minimised.

![YOLOv8s results](yolo_ultralytics_benchmark/results/images/yolov8s/results.png)
![YOLOv8s validation predictions](yolo_ultralytics_benchmark/results/images/yolov8s/val_batch0_pred.jpg)

---

#### YOLO11s

| Metric     | Value |
| --- | --- |
| mAP50      | 0.932 |
| mAP50-95   | 0.574 |
| Precision  | 0.892 |
| Recall     | 0.882 |
| Train time | 56.5 min |

Slightly below YOLOv8s in mAP50 but better mAP50-95 (0.574 vs 0.567), indicating marginally better localisation.

![YOLO11s results](yolo_ultralytics_benchmark/results/images/yolo11s/results.png)
![YOLO11s validation predictions](yolo_ultralytics_benchmark/results/images/yolo11s/val_batch0_pred.jpg)

---

#### YOLOv9s

| Metric     | Value |
| --- | --- |
| mAP50      | 0.925 |
| mAP50-95   | 0.559 |
| Precision  | 0.875 |
| Recall     | 0.856 |
| Train time | 89.9 min |

Weakest small model — takes nearly as long as the medium YOLOv9m (~95 min) but achieves considerably lower accuracy. YOLOv8s or YOLO11s are better choices at this tier.

![YOLOv9s results](yolo_ultralytics_benchmark/results/images/yolov9s/results.png)
![YOLOv9s validation predictions](yolo_ultralytics_benchmark/results/images/yolov9s/val_batch0_pred.jpg)

---

#### YOLO11n

| Metric     | Value |
| --- | --- |
| mAP50      | 0.907 |
| mAP50-95   | 0.524 |
| Precision  | 0.857 |
| Recall     | 0.838 |
| Train time | 54.2 min |

Best nano model alongside YOLOv10n. Fastest training of all models.

![YOLO11n results](yolo_ultralytics_benchmark/results/images/yolo11n/results.png)
![YOLO11n validation predictions](yolo_ultralytics_benchmark/results/images/yolo11n/val_batch0_pred.jpg)

---

#### YOLOv10n

| Metric     | Value |
| --- | --- |
| mAP50      | 0.907 |
| mAP50-95   | 0.525 |
| Precision  | 0.845 |
| Recall     | 0.854 |
| Train time | 63.1 min |

Ties YOLO11n in mAP50 but trains ~9 min slower. Marginally better mAP50-95.

![YOLOv10n results](yolo_ultralytics_benchmark/results/images/yolov10n/results.png)
![YOLOv10n validation predictions](yolo_ultralytics_benchmark/results/images/yolov10n/val_batch0_pred.jpg)

---

#### YOLO26n

| Metric     | Value |
| --- | --- |
| mAP50      | 0.887 |
| mAP50-95   | 0.508 |
| Precision  | 0.832 |
| Recall     | 0.831 |
| Train time | 70.8 min |

Weakest YOLO26 variant. Slower than YOLO11n while delivering lower accuracy — the larger YOLO26s/m are far more competitive.

![YOLO26n results](yolo_ultralytics_benchmark/results/images/yolo26n/results.png)
![YOLO26n validation predictions](yolo_ultralytics_benchmark/results/images/yolo26n/val_batch0_pred.jpg)

---

#### YOLOv9t

| Metric     | Value |
| --- | --- |
| mAP50      | 0.886 |
| mAP50-95   | 0.497 |
| Precision  | 0.840 |
| Recall     | 0.808 |
| Train time | 86.8 min |

Slowest and least accurate nano model. The tiny variant of YOLOv9 does not benefit from training time vs accuracy trade-off on this dataset.

![YOLOv9t results](yolo_ultralytics_benchmark/results/images/yolov9t/results.png)
![YOLOv9t validation predictions](yolo_ultralytics_benchmark/results/images/yolov9t/val_batch0_pred.jpg)

---

#### YOLOv8n — Fastest Training

| Metric     | Value |
| --- | --- |
| mAP50      | 0.886 |
| mAP50-95   | 0.500 |
| Precision  | 0.854 |
| Recall     | 0.794 |
| Train time | 50.6 min |

Fastest training time overall. Low recall (0.794) means it misses more turtles than other models — not ideal for conservation monitoring where false negatives carry high cost.

![YOLOv8n results](yolo_ultralytics_benchmark/results/images/yolov8n/results.png)
![YOLOv8n confusion matrix](yolo_ultralytics_benchmark/results/images/yolov8n/confusion_matrix_normalized.png)
![YOLOv8n validation predictions](yolo_ultralytics_benchmark/results/images/yolov8n/val_batch0_pred.jpg)

---

### Recommendations

| Use case | Recommended model |
| -------- | ----------------- |
| Best accuracy | **YOLOv9c** |
| Best accuracy + efficiency | **YOLO26m** or **YOLOv8m** |
| Resource-constrained deployment | **YOLOv8s** |
| Fastest iteration / prototyping | **YOLO11n** or **YOLOv8n** |
| Minimise missed detections (high recall) | **YOLO11m** or **YOLO26s** |

---

## Editing Colab Notebooks

1. Create a new branch on GitHub.
2. Open the notebook via:
   `https://colab.research.google.com/github/gab-es21/sea-turtles-detection/`
   (select your branch)
3. **File > Save a copy in Drive**, make edits, then **File > Save a copy in GitHub**.

## License

Dataset licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See [LICENSE](LICENSE) for repository license.
