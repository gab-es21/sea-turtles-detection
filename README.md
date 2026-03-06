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

## Editing Colab Notebooks

1. Create a new branch on GitHub.
2. Open the notebook via:
   `https://colab.research.google.com/github/gab-es21/sea-turtles-detection/`
   (select your branch)
3. **File > Save a copy in Drive**, make edits, then **File > Save a copy in GitHub**.

## License

Dataset licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See [LICENSE](LICENSE) for repository license.
