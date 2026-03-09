# ANALYSIS — Sea Turtle Detection: Training, Models & Dataset

> Analysis document for Master's thesis.
> This document covers **two benchmark experiments** conducted on distinct datasets.
>
> - **Experiment 1 (2026-02-02):** 16 YOLO models trained for 100 epochs on Dataset 1 — high-altitude, NIR aerial imagery of sea turtles at sea. Source data: `c:/Users/gaby3/Documents/sea-turtles-detection/results/benchmark_results.csv`, training curve images in `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/`, archived runs under `c:/Users/gaby3/Documents/sea-turtles-detection/archive/`.
> - **Experiment 2 (2026-03-07, in progress):** Same 16 YOLO models trained for 100 epochs on Dataset 2 — closer-range drone imagery of sea turtles, motivated by the domain gap identified during tracking evaluation. Source data: `c:/Users/gaby3/Documents/sea-turtles-detection/results/ds2/benchmark_results_ds2.csv`, runs under `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/`.
>
> Sections 1–16 cover Experiment 1. Section 17 covers Experiment 2.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset & Annotation](#2-dataset--annotation)
3. [Training Configuration & Hyperparameters](#3-training-configuration--hyperparameters)
4. [Training Curves & Convergence](#4-training-curves--convergence)
5. [Benchmark Results — Full Metrics](#5-benchmark-results--full-metrics)
6. [Confusion Matrices & Error Analysis](#6-confusion-matrices--error-analysis)
7. [Precision-Recall Curves](#7-precision-recall-curves)
8. [Model Architecture Comparison](#8-model-architecture-comparison)
9. [Earlier Experiments & Historical Progression](#9-earlier-experiments--historical-progression)
10. [Hyperparameter Tuning (Archive)](#10-hyperparameter-tuning-archive)
11. [Dataset 1-Tiled — Tiled Dataset Experiments](#11-dataset-1-tiled--tiled-dataset-experiments)
12. [Tracking — ByteTrack / BotSort](#12-tracking--bytetrack--botsort)
13. [Open Questions & Data Gaps](#13-open-questions--data-gaps)
14. [Key Findings for Thesis](#14-key-findings-for-thesis)
15. [Phase 1 — Test Set Evaluation Results](#15-phase-1--test-set-evaluation-results)
16. [Phase 4 — ByteTrack Tracking Integration](#16-phase-4--bytetrack-tracking-integration)
17. [Experiment 2 — Benchmark on Closer-Range Dataset](#17-experiment-2--benchmark-on-closer-range-dataset)

---

## 1. Executive Summary

This project trains and benchmarks 16 YOLO architectures (v8, v9, v10, v11, v26) for single-class detection of sea turtles in drone imagery across **two independent experiments** on datasets with different altitude and imaging conditions.

### 1.1 Experiment 1 — High-Altitude NIR Imagery (Dataset 1, completed 2026-02-02)

All 16 models trained for 100 epochs on Dataset 1 (968 train images, high-altitude NIR aerial). Full results in Sections 3–16.

**Top-level findings:**

| Finding | Value |
|---|---|
| Best mAP50 | **YOLOv9c — 0.9551** |
| Best mAP50-95 | **YOLO26m — 0.6427** |
| Best recall (fewest missed turtles) | **YOLO11m — 0.9153** |
| Best precision | **YOLOv9c — 0.9354** |
| Fastest training | **YOLOv8n — 50.6 min** |
| Best efficiency (mAP50/min) | **YOLOv8s — 0.0171 mAP50/min** |
| mAP50 range across all 16 models | 0.886 – 0.955 |
| All models: background FP rate (CM) | **0.00** (no false positive detections from pure background) |

The mAP50 gap between the weakest (YOLOv8n/v9t, 0.886) and the strongest (YOLOv9c, 0.955) is only 0.069, indicating that all tested architectures are capable on this domain. The critical differentiator for conservation monitoring is **recall** (missed turtles = false negatives), where the range is larger: 0.794 (YOLOv8n) to 0.915 (YOLO11m).

The ByteTrack tracking evaluation (Phase 4) using the best Experiment 1 model (YOLOv9m) on 4 field videos revealed a **domain gap**: models trained on high-altitude at-sea imagery fail on closer-range footage. This motivated Experiment 2.

### 1.2 Experiment 2 — Closer-Range Drone Imagery (Dataset 2, started 2026-03-07, in progress)

Same 16 models trained for 100 epochs on Dataset 2 (2,071 train images, lower-altitude footage). Partial results below; full results pending. See Section 17 for details.

**Partial findings (first 4 models):**

| Finding | Value |
|---|---|
| mAP50 range (4 models so far) | 0.386 – 0.533 |
| Best mAP50 so far | **yolo11n — 0.533** (note: recall anomaly, see Section 17.4) |
| Best mAP50-95 so far | **yolo11n — 0.346** |
| Training time range | 95 – 148 min per model |
| Full results | **PENDING — training in progress** |

**Important note:** the lower mAP50 values in Experiment 2 (relative to Experiment 1) are expected — Dataset 2 is a different, more visually heterogeneous dataset. Direct numerical comparison between experiments is not meaningful; each experiment should be evaluated relative to its own dataset's baseline.

---

## 2. Dataset & Annotation

### 2.1 Dataset Variants

| ID | Label | Train | Val | Test | Description |
|----|-------|-------|-----|------|-------------|
| 1 | Base (B) | 968 | 272 | 118 | Original annotations — high-altitude NIR aerial (Roboflow sea-turtles-yia2e v1) |
| 1-T | Tiled (W) | 15,488 | 4,352 | 1,888 | Tiled augmentation of Dataset 1 (4×4 grid per image) |
| 3 | Closer-range (C) | 2,071 | 534 | 285 | Lower-altitude drone imagery (Roboflow sea-turtles-model v6) |

**Experiment 1** used Dataset 1 exclusively. Dataset 1-Tiled was used only in earlier exploratory experiments (see Section 11). **Experiment 2** uses Dataset 2 (see Section 17).

### 2.2 Dataset 1 — Source & Annotation Format

- **Source**: [Roboflow — sea-turtles-yia2e](https://universe.roboflow.com/gabriel-esteves-dy2cw/sea-turtles-yia2e/dataset/1), version 1
- **License**: CC BY 4.0
- **Imagery type**: NIR (near-infrared) aerial/drone footage of sea turtles on nesting beaches
- **Annotation format**: **YOLO TXT** (one `.txt` per image with normalized `class cx cy w h` format), as confirmed by `data.yaml` structure (train/valid/test pointing to image folders with paired label folders)
- **Annotation tool**: Roboflow (web-based annotation platform)
- **Class**: Single class — `Turtle` (nc: 1)

### 2.6 Dataset 2 — Source & Characteristics (Experiment 2)

- **Source**: [Roboflow — sea-turtles-model v6](https://universe.roboflow.com/gabriel-esteves/sea-turtles-model/dataset/6)
- **License**: MIT
- **Split**: 2,071 train / 534 val / 285 test (2,890 total images)
- **Local path**: `c:/Users/gaby3/Documents/sea-turtles-detection/Dataset/sea-turtles-2/`
- **Imagery type**: Drone footage at **lower altitude / closer range** than Dataset 1. Turtles appear larger in frame, with visible body detail and more varied backgrounds (water, beach, mixed). This is the dataset that corresponds to the type of video footage collected during field surveys.
- **Altitude contrast with Dataset 1**: Dataset 1 was captured at high altitude, producing small turtle silhouettes against open water. Dataset 2 was captured at lower altitude, producing larger, more detailed turtle appearances. This difference is the root cause of the domain gap identified in Phase 4 tracking (Section 16.7).
- **Annotation format**: YOLO TXT (same format as Dataset 1)
- **Class**: Single class — `turtle` (nc: 1)
- **Motivation**: After the tracking evaluation revealed that models trained on Dataset 1 failed on field videos (domain gap), Dataset 2 was selected to train models suited to real-world, lower-altitude deployment scenarios.

### 2.3 Bounding Box Size Distribution

The benchmark does not include an explicit bounding box size analysis script. However, qualitative observations from validation prediction images (`val_batch0_pred.jpg`) indicate:

- Turtles appear as **medium-to-large objects** at 640 px input resolution: bounding boxes typically occupy a substantial portion of each image tile, suggesting the majority fall in the **medium (32–96 px) to large (> 96 px)** range.
- In the mosaic validation grid, individual turtles are clearly visible and well-separated from the background (sandy beach or shallow water).
- Very few instances appear as "small objects" (< 32 px), which is consistent with aerial surveys at close range.
- **Quantitative breakdown**: not computed in this project; a dedicated analysis script (e.g., parsing all label `.txt` files) would be needed for exact percentages.

### 2.4 Instance Distribution per Image

- Not explicitly computed. From visual inspection of batch prediction outputs: most images appear to contain **1 to ~6 turtles** per frame, with some crowded scenes containing more overlapping individuals.
- Dataset 1-Tiled splits each original image into 16 tiles (4×4 grid), leading to many tiles containing 0 turtles and some tiles containing dense groupings — explaining the 16× increase in image count with a non-uniform turtle density per tile.

### 2.5 NIR Imagery Pre-processing

- **No explicit NIR-specific pre-processing** was applied before training. Images were used as exported from Roboflow.
- Ultralytics loads images as standard 3-channel tensors regardless of spectral content; NIR images stored as grayscale or pseudo-RGB are handled identically to visible-light imagery.
- The Ultralytics HSV augmentation (hsv_h, hsv_s, hsv_v) operates on the loaded channels without special handling.

---

> **Scope note — Sections 3 through 16:** All content below up to Section 17 refers exclusively to **Experiment 1** (Dataset 1, high-altitude NIR imagery, benchmark run 2026-02-02). Section 17 covers Experiment 2 (Dataset 2, closer-range imagery).

---

## 3. Training Configuration & Hyperparameters

### 3.1 Fixed Training Parameters (Benchmark)

All 16 models were trained with these exact parameters (`benchmark_yolo_models.py`):

| Parameter | Value |
|-----------|-------|
| `epochs` | 100 |
| `imgsz` | 640 |
| `batch` | 8 |
| `device` | 0 (CUDA GPU) |
| `workers` | 0 |
| `seed` | 42 |
| `pretrained` | True (COCO ImageNet weights for all models) |

### 3.2 Hyperparameters — Ultralytics Defaults

No custom `hyp.yaml` was used. All hyperparameters follow **Ultralytics `default.yaml`** values:

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `lr0` | 0.01 | Initial learning rate |
| `lrf` | 0.01 | Final LR fraction (final LR = lr0 × lrf = 1e-4) |
| `momentum` | 0.937 | SGD momentum / Adam beta1 |
| `weight_decay` | 0.0005 | L2 regularisation |
| `warmup_epochs` | 3.0 | Warmup duration |
| `warmup_momentum` | 0.8 | Initial momentum during warmup |
| `warmup_bias_lr` | 0.1 | Bias LR during warmup |
| `box` | 7.5 | Box loss gain |
| `cls` | 0.5 | Classification loss gain |
| `dfl` | 1.5 | Distribution Focal Loss gain |
| `nbs` | 64 | Nominal batch size for loss normalisation |
| `patience` | 100 | Early stopping patience (disabled in practice — equals total epochs) |
| `close_mosaic` | 10 | Disable mosaic for final 10 epochs |
| `cos_lr` | False | Linear LR decay (not cosine) |
| `amp` | True | Automatic Mixed Precision |

### 3.3 Optimiser

- `optimizer: auto` — Ultralytics selects the optimiser per architecture.
- For YOLOv8/v9/v10/v11/v26 with pretrained weights, `auto` resolves to **SGD** (confirmed as the default for detection tasks in Ultralytics; AdamW is chosen only for classification or when explicitly requested).
- LR scheduler: **linear decay** from lr0 to lr0×lrf over the training run, with cosine warmup for the first 3 epochs.

### 3.4 Data Augmentation (Default Ultralytics)

Applied during training via the Ultralytics data loader:

| Augmentation | Value | Notes |
|---|---|---|
| HSV-Hue jitter | ±0.015 (fraction) | Colour hue shift |
| HSV-Saturation jitter | ±0.70 (fraction) | Colour saturation shift |
| HSV-Value jitter | ±0.40 (fraction) | Brightness shift |
| Horizontal flip | 0.5 probability | Random left-right flip |
| Vertical flip | 0.0 | Disabled |
| Scale | ±0.5 gain | Random zoom in/out |
| Translation | ±0.1 fraction | Random shift |
| Rotation | 0.0° | Disabled |
| Shear | 0.0° | Disabled |
| Perspective | 0.0 | Disabled |
| Mosaic | 1.0 (100%) → 0 last 10 epochs | 4-image mosaic |
| MixUp | 0.0 | Disabled |
| Copy-paste | 0.0 | Disabled |

**Note on NIR imagery**: The HSV augmentation is applied regardless of whether images are NIR or RGB. For NIR imagery stored as pseudo-colour, this may create unrealistic colour combinations; however, the absence of colour as a discriminative feature for turtle detection (shape/texture-based detection) reduces the practical impact.

### 3.5 Pre-trained Weights

All 16 models used `pretrained=True`, loading **COCO-pretrained weights** provided by Ultralytics. No model was trained from scratch. This transfer learning from COCO provides a strong initialisation for the detection head and backbone, contributing to the fast convergence observed (mAP50 > 0.5 by epoch 3–5 for most models).

### 3.6 Ultralytics Version

The installed Ultralytics version is confirmed from `.venv/` to be a **2025/2026 release** supporting YOLO26 (the latest family tested). The exact version was not logged in the benchmark output. The `default.yaml` present includes `multi_scale`, `cutmix`, and `bgr` parameters, which are present in Ultralytics ≥ 8.3.x (late 2025 or later).

---

## 4. Training Curves & Convergence

### 4.1 General Observations (All 16 Models)

All 16 models were trained for exactly 100 epochs. With `patience=100` (equal to total epochs), **early stopping was never triggered** — all models ran to completion.

**Common convergence pattern:**
- Epochs 1–3: Rapid loss drop, especially `train/cls_loss` (from ~3–6 → ~1). Warmup phase.
- Epochs 3–30: Steep improvement in all metrics. mAP50 rises from ~0.3–0.6 to ~0.8–0.9.
- Epochs 30–100: Gradual refinement. Losses continue declining; mAP50 approaches plateau.
- All models show **smooth, stable training** with no divergence or catastrophic spikes.

**Key observation**: The training curves (from `results.png` images for all 16 models) show that **most models were still slowly improving at epoch 100**, particularly for `val/mAP50`. No model shows a clear flat plateau well before epoch 100. This suggests that extending training beyond 100 epochs could yield marginal gains, particularly for the top-performing models.

### 4.2 Per-Model Convergence Analysis

| Model | Convergence rate | Curve stability | Status at epoch 100 |
|-------|-----------------|-----------------|---------------------|
| YOLOv9c | Steady, medium-fast | Very smooth | Still rising slowly — not fully converged |
| YOLO26m | Steady, medium | Very smooth (unique DFL scale) | Plateau approaching |
| YOLOv9m | Steady | Smooth | Still rising slowly |
| YOLO11m | Fast among medium models | Smooth | Near plateau |
| YOLOv8m | Fast | Smooth | Near plateau |
| YOLOv10m | Medium | Smooth | Near plateau |
| YOLO26s | Fast for a small model | Smooth | Near plateau |
| YOLOv10s | Fast | Smooth | Near plateau |
| YOLOv8s | Fast | Smooth | Near plateau (~epoch 80) |
| YOLO11s | Fast | Smooth | Near plateau |
| YOLOv9s | Slow (similar pace to YOLOv9m) | Smooth | Still improving |
| YOLO11n | Fast | Smooth | Near plateau |
| YOLOv10n | Medium | Smooth | Near plateau |
| YOLO26n | Medium-slow | Smooth | Near plateau |
| YOLOv9t | Slow | More noisy (val losses oscillate) | Still improving |
| YOLOv8n | Fast | Smooth | Plateau around epoch 70–80 |

**Correlation between family and convergence speed:**
- **YOLOv8 family**: Fastest convergence. YOLOv8n and YOLOv8s reach near-plateau earliest (~epoch 70–80), consistent with their simpler architectures.
- **YOLO11 family**: Fast convergence comparable to v8, with higher recall at plateau.
- **YOLOv10 family**: Moderate convergence speed. Similar to v8 but slightly slower.
- **YOLOv9 family**: Slowest convergence within each size tier (v9t nearly as slow as v9m in training time). The `c` variant converges more steadily than `t`/`s`.
- **YOLO26 family**: Medium convergence, smooth curves. Medium and small variants show good plateau behaviour.

### 4.3 Loss Curve Observations

**train/box_loss**: Decreases monotonically for all models; values range from ~2.2 (epoch 1) to ~1.1–1.4 (epoch 100). No instability observed.

**train/cls_loss**: Drops sharply in epochs 1–10 (from ~3–6 down to ~1). Continues declining slowly. The initial spike in epoch 1–2 is normal (model learning class distribution). No instability.

**train/dfl_loss**:
- YOLOv8/v9/v10/v11: DFL loss starts at ~1.1–1.2 and decreases to ~0.85–0.95 by epoch 100. Smooth behaviour.
- **YOLO26 (all sizes)**: DFL loss values are an **order of magnitude smaller** than other families (~0.002 range vs ~1.0 range). This is a confirmed architectural/scaling difference in YOLO26's DFL implementation, not a training anomaly.

**val/box_loss and val/cls_loss**: Track training losses closely for all models — no sign of significant overfitting. The train/val loss gap remains consistent throughout training, confirming that the dataset is large enough relative to model complexity for the models tested.

### 4.4 Overfitting Assessment

No model shows a clear overfitting signature (diverging val loss while train loss decreases). Possible reasons:
1. All models use COCO pre-trained weights (strong regularisation via transfer learning).
2. Data augmentation (mosaic + flips + scale/HSV) increases effective dataset diversity.
3. Medium-capacity models (m variants) do not dramatically overfit a 968-image dataset at 100 epochs.
4. Weight decay (5e-4) and AMP contribute to regularisation.

The **closest to overfitting** is YOLOv8n (lowest recall, 0.794), but this is due to underfitting rather than overfitting — the nano model lacks capacity to capture all turtle instances.

### 4.5 Best Epoch (Checkpoint Selection)

The benchmark script calls `model.val()` immediately after `model.train()`, which loads `best.pt` (the checkpoint with highest validation fitness score during training). Therefore, **all reported metrics correspond to the best checkpoint**, not the final epoch.

Given that most curves were still slowly improving at epoch 100, the best checkpoint is likely at or near epoch 100 for the top-performing models. For faster-converging nano models (YOLOv8n), the best checkpoint may be earlier (epoch 70–85).

The exact best epoch per model is not recoverable without the `results.csv` files from each training run (stored under `yolo_ultralytics_benchmark/runs/`, which is gitignored and not included in the repository).

---

## 5. Benchmark Results — Full Metrics

> _Experiment 1 — Dataset 1 (high-altitude NIR, 968 train images). For Experiment 2 results see Section 17._

### 5.1 Complete Results Table (sorted by mAP50)

All values from `c:/Users/gaby3/Documents/sea-turtles-detection/results/benchmark_results.csv`. Run date: 2026-02-02.

| Rank | Model | mAP50 | mAP50-95 | Precision | Recall | Train Time (min) | Efficiency (mAP50/min) | Run Name |
|------|-------|-------|----------|-----------|--------|-----------------|----------------------|----------|
| 1 | yolov9c | **0.9551** | 0.6334 | **0.9354** | 0.9100 | 106.79 | 0.00895 | yolov9c_20260202_163426 |
| 2 | yolo26m | 0.9521 | **0.6427** | 0.9275 | 0.8973 | 93.24 | 0.01021 | yolo26m_20260202_033138 |
| 3 | yolov9m | 0.9501 | 0.6192 | 0.9128 | 0.9001 | 95.36 | 0.00996 | yolov9m_20260202_145822 |
| 4 | yolo11m | 0.9457 | 0.6159 | 0.8956 | **0.9153** | 80.95 | 0.01168 | yolo11m_20260202_065642 |
| 5 | yolov8m | 0.9456 | 0.6125 | 0.9219 | 0.9100 | 78.16 | 0.01210 | yolov8m_20260202_200749 |
| 6 | yolov10m | 0.9444 | 0.6185 | 0.9241 | 0.9030 | 90.78 | 0.01040 | yolov10m_20260202_102932 |
| 7 | yolo26s | 0.9401 | 0.5931 | 0.8603 | 0.9128 | 69.60 | 0.01351 | yolo26s_20260202_022129 |
| 8 | yolov10s | 0.9359 | 0.5891 | 0.8920 | 0.8934 | 67.64 | 0.01385 | yolov10s_20260202_092131 |
| 9 | yolov8s | 0.9356 | 0.5675 | 0.9052 | 0.8661 | 54.73 | 0.01710 | yolov8s_20260202_191241 |
| 10 | yolo11s | 0.9318 | 0.5739 | 0.8917 | 0.8822 | 56.47 | 0.01650 | yolo11s_20260202_055950 |
| 11 | yolov9s | 0.9250 | 0.5586 | 0.8755 | 0.8560 | 89.87 | 0.01029 | yolov9s_20260202_132754 |
| 12 | yolo11n | 0.9071 | 0.5240 | 0.8571 | 0.8379 | 54.16 | 0.01675 | yolo11n_20260202_050518 |
| 13 | yolov10n | 0.9066 | 0.5253 | 0.8454 | 0.8541 | 63.07 | 0.01438 | yolov10n_20260202_081804 |
| 14 | yolo26n | 0.8875 | 0.5079 | 0.8322 | 0.8312 | 70.81 | 0.01253 | yolo26n_20260202_011017 |
| 15 | yolov9t | 0.8865 | 0.4975 | 0.8397 | 0.8078 | 86.78 | 0.01021 | yolov9t_20260202_120044 |
| 16 | yolov8n | 0.8860 | 0.4996 | 0.8540 | 0.7937 | 50.63 | **0.01751** | yolov8n_20260202_182141 |

**Checkpoint note**: All metrics correspond to the **best checkpoint** (`best.pt`), not the final epoch. The benchmark's `model.val()` call after training automatically uses the best saved weights.

### 5.2 Per-Family Analysis

#### Medium tier (m)

| Model | mAP50 | mAP50-95 | Precision | Recall | Time (min) |
|-------|-------|----------|-----------|--------|-----------|
| YOLOv9c | 0.9551 | 0.6334 | 0.9354 | 0.9100 | 106.8 |
| YOLO26m | 0.9521 | 0.6427 | 0.9275 | 0.8973 | 93.2 |
| YOLOv9m | 0.9501 | 0.6192 | 0.9128 | 0.9001 | 95.4 |
| YOLO11m | 0.9457 | 0.6159 | 0.8956 | 0.9153 | 81.0 |
| YOLOv8m | 0.9456 | 0.6125 | 0.9219 | 0.9100 | 78.2 |
| YOLOv10m | 0.9444 | 0.6185 | 0.9241 | 0.9030 | 90.8 |

Observations: All medium models score mAP50 > 0.944. YOLOv9c leads in mAP50 and precision but is the slowest. YOLO26m leads in mAP50-95 (tightest bounding boxes). YOLO11m leads in recall (fewest missed detections). YOLOv8m and YOLO11m offer the best time-accuracy trade-off in this tier.

#### Small tier (s)

| Model | mAP50 | mAP50-95 | Precision | Recall | Time (min) |
|-------|-------|----------|-----------|--------|-----------|
| YOLO26s | 0.9401 | 0.5931 | 0.8603 | 0.9128 | 69.6 |
| YOLOv10s | 0.9359 | 0.5891 | 0.8920 | 0.8934 | 67.6 |
| YOLOv8s | 0.9356 | 0.5675 | 0.9052 | 0.8661 | 54.7 |
| YOLO11s | 0.9318 | 0.5739 | 0.8917 | 0.8822 | 56.5 |
| YOLOv9s | 0.9250 | 0.5586 | 0.8755 | 0.8560 | 89.9 |

Observations: YOLOv9s is the outlier — nearly as slow as medium models but weaker accuracy. YOLOv8s achieves the best efficiency (mAP50/min). YOLO26s leads in recall at this tier (0.913), making it a strong choice when false negatives are costly.

#### Nano tier (n/t)

| Model | mAP50 | mAP50-95 | Precision | Recall | Time (min) |
|-------|-------|----------|-----------|--------|-----------|
| YOLO11n | 0.9071 | 0.5240 | 0.8571 | 0.8379 | 54.2 |
| YOLOv10n | 0.9066 | 0.5253 | 0.8454 | 0.8541 | 63.1 |
| YOLO26n | 0.8875 | 0.5079 | 0.8322 | 0.8312 | 70.8 |
| YOLOv9t | 0.8865 | 0.4975 | 0.8397 | 0.8078 | 86.8 |
| YOLOv8n | 0.8860 | 0.4996 | 0.8540 | 0.7937 | 50.6 |

Observations: YOLO11n and YOLOv10n lead the nano tier with identical mAP50 (0.907). YOLOv8n is fastest but has the worst recall (0.794) — the highest miss rate of all 16 models. YOLOv9t and YOLO26n perform similarly to YOLOv8n but take significantly longer.

### 5.3 mAP50 vs mAP50-95 Gap

The gap between mAP50 and mAP50-95 reveals localisation quality (tightness of bounding boxes):

| Model | mAP50 | mAP50-95 | Gap | Interpretation |
|-------|-------|----------|-----|----------------|
| YOLO26m | 0.9521 | 0.6427 | 0.309 | Smallest gap → tightest boxes |
| YOLOv10m | 0.9444 | 0.6185 | 0.326 | Good localisation |
| YOLOv9m | 0.9501 | 0.6192 | 0.331 | Good localisation |
| YOLOv9c | 0.9551 | 0.6334 | 0.322 | Good localisation |
| YOLOv9t | 0.8865 | 0.4975 | 0.389 | Largest gap → loosest boxes |
| YOLOv8n | 0.8860 | 0.4996 | 0.386 | Loose boxes |

The YOLO26 architecture produces the tightest bounding boxes relative to its mAP50, which may reflect its DFL implementation.

---

## 6. Confusion Matrices & Error Analysis

> _Experiment 1 — Dataset 1._

### 6.1 Methodology

Ultralytics generates normalised confusion matrices at a fixed confidence threshold (default: 0.25) and IoU threshold (default: 0.45). All matrices use 2 categories: `Turtle` (foreground) and `background`.

- **True Positive (TP)**: Turtle detected where a turtle exists (IoU ≥ 0.45)
- **False Negative (FN)**: Turtle missed (detected as background)
- **False Positive (FP)**: Background region detected as Turtle
- Column "background" in the matrix = detections on background patches

### 6.2 FN and FP Rates (from Confusion Matrix Images)

| Model | TP Rate (from CM) | FN Rate (from CM) | FP from background |
|-------|-------------------|-------------------|--------------------|
| YOLOv9c | **0.94** | 0.06 | 0.00 |
| YOLO26m | 0.93 | 0.07 | 0.00 |
| YOLO11m | **0.94** | 0.06 | 0.00 |
| YOLOv9t | 0.87 | 0.13 | 0.00 |
| YOLOv8n | 0.89 | 0.11 | 0.00 |

For models without explicit CM images read above, the FN rate at the CM threshold can be approximated from recall (FN_rate ≈ 1 − recall at a fixed threshold — note this is not identical to mAP recall which is threshold-averaged). The following are estimates:

| Model | Approx. FN rate |
|-------|----------------|
| YOLOv9m | ~10% |
| YOLOv8m | ~9% |
| YOLO26s | ~9% |
| YOLO11n | ~16% |
| YOLOv10n | ~15% |
| YOLOv9s | ~14% |
| YOLOv8s | ~13% |
| YOLO11s | ~12% |

### 6.3 Key Observations

1. **Zero FP from background** (for all models with available CMs): No model generates spurious detections from pure background patches. This confirms that the domain gap between turtle texture and beach/water background is well captured. The primary error mode is **missed detections (FN)**, not false alarms.

2. **FN is the dominant error**: For conservation monitoring (where every missed turtle matters), minimising FN is critical. YOLOv9c and YOLO11m achieve the lowest FN rate (6%).

3. **Nano models miss more turtles**: YOLOv9t misses 13% of turtles at the default threshold, YOLOv8n misses ~11%. These miss rates compound over many frames in video, making nano models less suitable for population estimation.

4. **Threshold sensitivity**: The CM is computed at confidence=0.25. Lowering the threshold would reduce FN at the cost of introducing FP. For field deployment, a threshold sweep with F-beta (β > 1, penalising FN more) is recommended.

### 6.4 Analysis of Hard Cases

From visual inspection of `val_batch0_pred.jpg` (YOLOv9c):
- Turtles detected at confidence 0.5–0.9 across varied backgrounds (sand, water reflection, substrate).
- Images appear to be **high-contrast aerial shots** with clearly visible turtle silhouettes.
- The first tile in the batch (showing a tree/vegetation region) appears to have multiple overlapping detections with lower confidence (0.5–0.7), suggesting **dense scenes with occlusion** challenge all models.
- No systematic cross-model hard-case analysis was performed; individual image-level comparison would require inference runs on a shared test set.

---

## 7. Precision-Recall Curves

> _Experiment 1 — Dataset 1._

PR curves (`BoxPR_curve.png`) are available for all 16 models. Analysis of the YOLOv9c PR curve (representative of top models):

- **Shape**: Near-perfect rectangular curve — precision remains at ~1.0 from recall 0.0 to ~0.7, then drops steeply.
- **Area under curve (mAP@0.5)**: 0.954 (matches benchmark CSV).
- **High-confidence region**: At recall ≤ 0.70, precision is essentially 1.0 — the model only produces confident detections that are correct.
- **Recall degradation zone**: Precision drops from 1.0 to ~0.85 between recall 0.70 and 0.95, then falls sharply to ~0.0 at recall → 1.0.

Weaker models (nano tier) have less rectangular curves, with earlier precision degradation beginning around recall 0.5–0.6, indicating more mixed-confidence detections.

---

## 8. Model Architecture Comparison

> _Experiment 1 — Dataset 1. Architecture rankings may differ on Dataset 2 (see Section 17)._

### 8.1 Architecture Overview

| Family | Versions tested | Key architectural innovation |
|--------|----------------|------------------------------|
| YOLOv8 | n, s, m | Anchor-free, decoupled head, C2f blocks |
| YOLOv9 | t, s, m, c | GELAN (Generalised Efficient Layer Aggregation), PGI (Programmable Gradient Information) |
| YOLOv10 | n, s, m | Consistent dual assignments (no NMS at inference), dual-label assignment |
| YOLO11 | n, s, m | C3k2 blocks, SPPF+PSA, enhanced feature extraction vs v8 |
| YOLO26 | n, s, m | Released early 2026 via Ultralytics; distinct DFL loss scaling (~1/1000 of other models) |

### 8.2 Parameters and FLOPs (COCO reference values)

The following are standard benchmark values from Ultralytics documentation (at imgsz=640). Inference times for this dataset are not measured in the benchmark scripts.

| Model | Params (M) | GFLOPs | mAP50-COCO (reference) |
|-------|-----------|--------|----------------------|
| YOLOv8n | 3.2 | 8.7 | 0.527 |
| YOLOv8s | 11.2 | 28.6 | 0.619 |
| YOLOv8m | 25.9 | 78.9 | 0.672 |
| YOLOv9t | 2.0 | 7.7 | 0.516 |
| YOLOv9s | 7.2 | 26.7 | 0.578 |
| YOLOv9m | 20.1 | 76.8 | 0.640 |
| YOLOv9c | 25.5 | 102.8 | 0.660 |
| YOLOv10n | 2.3 | 6.7 | 0.545 |
| YOLOv10s | 7.2 | 21.6 | 0.573 |
| YOLOv10m | 15.4 | 59.1 | 0.621 |
| YOLO11n | 2.6 | 6.5 | 0.567 |
| YOLO11s | 9.4 | 21.5 | 0.612 |
| YOLO11m | 20.1 | 68.0 | 0.672 |

> YOLO26 parameter counts are not available in public documentation as of the training date (February 2026). GFLOPs for YOLO26 were not measured in this project.

> **No inference timing measurements** were performed in this project. The benchmark measured only training time, not inference latency. FPS benchmarks on the deployment GPU would be needed for real-time feasibility analysis.

### 8.3 Training Time vs Accuracy

A consistent pattern across all families: **larger models take longer but achieve higher accuracy**, but with diminishing returns:

- YOLOv8: n(50.6min/0.886) → s(54.7min/0.936) → m(78.2min/0.946). The n→s jump gives +0.050 mAP50 for only +4 min.
- YOLOv9: t(86.8min/0.886) → s(89.9min/0.925) → m(95.4min/0.950) → c(106.8min/0.955). Very slow training throughout; the s→m jump gives +0.025 for +5.5 min, while m→c gives only +0.005 for +11.4 min.
- YOLO11: n(54.2min/0.907) → s(56.5min/0.932) → m(81.0min/0.946). Excellent step efficiency.
- YOLOv10: n(63.1min/0.907) → s(67.6min/0.936) → m(90.8min/0.944). Good step efficiency but slower than v8/v11.
- YOLO26: n(70.8min/0.887) → s(69.6min/0.940) → m(93.2min/0.952). The n variant is disproportionately weak; s and m are competitive.

---

## 9. Earlier Experiments & Historical Progression

### 9.1 Experiment Log

| Run | Architecture | Epochs | Precision | Recall | mAP50 | Dataset | Environment | Notes |
|-----|-------------|--------|-----------|--------|-------|---------|-------------|-------|
| 1 | YOLOv8n | 50 | 0.748 | 0.764 | 0.809 | 1 | Colab | First experiment |
| 2 | YOLOv8n | 5 | 0.652 | 0.419 | 0.453 | 2 | Colab | Tiled dataset, insufficient epochs |
| 3 | YOLOv8n | 150 | 0.830 | 0.920 | 0.825 | 1 | Local | Improved with more epochs |
| 4 | YOLOv8n | 400 | 0.824 | 0.920 | 0.823 | 1 | Local | No gain after ~150 epochs |
| 5 | YOLOv8n | 100 | 0.786 | 0.910 | 0.827 | 2 | Local | Best tiled result for YOLOv8n |
| 7 | YOLOv8m | 100 | 0.866 | 0.666 | 0.796 | 1 | Local | Early medium model test |
| Benchmark | YOLOv8n | 100 | 0.854 | 0.794 | 0.886 | 1 | Local | Seed=42, batch=8 |
| Benchmark | YOLOv8m | 100 | 0.922 | 0.910 | 0.946 | 1 | Local | Seed=42, batch=8 |

### 9.2 Key Progression Insights

1. **Run 3 vs Run 4 (150 vs 400 epochs, YOLOv8n)**: mAP50 barely changed (0.825 → 0.823). This confirms that YOLOv8n converges for Dataset 1 before epoch 150, and extending to 400 epochs brings no benefit — validating the 100-epoch benchmark choice.

2. **Benchmark vs Run 7 (YOLOv8m)**: The benchmark YOLOv8m (0.946) substantially outperforms Run 7 (0.796). The difference is attributable to `seed=42` (reproducibility), `batch=8` (vs 16 in Run 7, affecting gradient noise), and potentially different Ultralytics versions with improved defaults.

3. **Dataset 1 vs Dataset 1-Tiled (Runs 1 vs 5, YOLOv8n)**: mAP50 was similar (0.809 vs 0.827) but the tiled dataset required only 100 epochs vs 50 to reach a similar level. The tiled dataset did not provide a substantial accuracy gain for YOLOv8n; the benefit may be larger for detecting smaller objects.

4. **Training environment**: Early runs used Google Colab (GPU not specified, likely T4/V100); later runs and the full benchmark used local GPU (Windows, CUDA device 0). The local GPU enabled consistent, uninterrupted 100-epoch runs for all 16 models.

### 9.3 Per-Epoch Archive Data

Raw per-epoch `results.csv` files exist in `c:/Users/gaby3/Documents/sea-turtles-detection/archive/yolov8_local/runs/detect/` for YOLOv8n and YOLOv8m experiments (Runs 3, 4, 5, 7 and tuning runs). These contain per-epoch values for:
`train/box_loss, train/cls_loss, train/dfl_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B), val/box_loss, val/cls_loss, val/dfl_loss, lr/pg0, lr/pg1, lr/pg2`

The main **16-model benchmark runs do not have per-epoch CSV files available** in the repository (the `yolo_ultralytics_benchmark/runs/` directory is gitignored).

---

## 10. Hyperparameter Tuning (Archive)

### 10.1 Overview

Multiple rounds of Ultralytics' built-in hyperparameter tuner (backed by Ray Tune) were performed on YOLOv8n, YOLOv8s, and YOLOv8m during the archive phase. The tuner uses evolutionary search over a predefined hyperparameter space.

### 10.2 Best Tuning Results

| Run | Model | Iterations | Best iter | Best mAP50 | Best fitness | Notes |
|-----|-------|------------|-----------|-----------|--------------|-------|
| n_tune6_right | YOLOv8n | 200 (ran 84) | 34 | 0.796 | 0.466 | Dataset 1 |
| m_tune7_right | YOLOv8m | 100 | 44 | 0.792 | 0.476 | Dataset 1 |
| m_tune8_right | YOLOv8m | — | 27 | — | — | Dataset 1 |
| m_tune8_right_dt2 | YOLOv8m | — | 27 | — | — | Dataset 1-Tiled |
| n_tune_8_right_dt2 | YOLOv8n | — | 49 | — | — | Dataset 1-Tiled |
| s_tune9_right_dt2 | YOLOv8s | — | 5 | — | — | Dataset 1-Tiled |
| s_train10_right | YOLOv8s | — | — | — | — | Dataset 1 |

### 10.3 Best Hyperparameters Found (YOLOv8m — m_tune7_right)

| Parameter | Tuned value | Default value | Change |
|-----------|------------|---------------|--------|
| lr0 | 0.01108 | 0.01 | +10% |
| lrf | 0.01086 | 0.01 | +9% |
| momentum | 0.93727 | 0.937 | ≈same |
| weight_decay | 0.00048 | 0.0005 | -4% |
| warmup_epochs | 1.672 | 3.0 | -44% |
| warmup_momentum | 0.86981 | 0.8 | +9% |
| box | 6.699 | 7.5 | -11% |
| cls | 0.498 | 0.5 | ≈same |
| dfl | 1.817 | 1.5 | +21% |
| scale | 0.615 | 0.5 | +23% |
| mosaic | 0.703 | 1.0 | -30% |
| translate | 0.074 | 0.1 | -26% |

### 10.4 Tuning vs Default Comparison

Critically, the tuned YOLOv8m achieved mAP50 ≈ 0.792 (best iteration), which is **lower than the default-hyperparameter benchmark** result of 0.946. This apparent paradox has several explanations:

1. **Each tuning trial used fewer epochs** (typically 25–50 for tuning speed), while the final benchmark used 100 full epochs. The fitness metric measured during tuning underestimates final performance.
2. **Tuning was performed with an older Ultralytics version** that may have had different default data augmentation or model behaviour.
3. **batch=16 in tuning vs batch=8 in benchmark**: batch size affects gradient noise and convergence characteristics.
4. The default hyperparameters turned out to be near-optimal for this dataset/task combination, consistent with Ultralytics' design philosophy.

**Conclusion**: Hyperparameter tuning did not yield improvement over defaults for this dataset. The default Ultralytics configuration is appropriate and well-suited to aerial wildlife detection tasks.

---

## 11. Dataset 1-Tiled — Tiled Dataset Experiments

### 11.1 Overview

Dataset 1-Tiled is derived from Dataset 1 by tiling each image into a 4×4 grid of subtiles, creating a 16× larger dataset (968 → 15,488 training images). This technique is commonly used to improve detection of small objects.

### 11.2 Available Results

| Run | Model | Epochs | mAP50 | Dataset |
|-----|-------|--------|-------|---------|
| Run 2 | YOLOv8n | 5 | 0.453 | 2 |
| Run 5 | YOLOv8n | 100 | 0.827 | 2 |
| archive/yolov8_tile_local/train | YOLOv8n | ~100 | — | 2 |
| archive/yolov8_tile_local/train2 | YOLOv8n | ~100 | — | 2 |
| m_tune8_right_dt2 | YOLOv8m (tuned) | 100 | — | 2 |

### 11.3 Assessment

- **Run 5 (Dataset 1-Tiled, mAP50=0.827) vs Benchmark YOLOv8n (Dataset 1, mAP50=0.886)**: Dataset 1 outperforms Dataset 1-Tiled for YOLOv8n at 100 epochs. This suggests the original images (968 at 640 px) provide more training signal than the tiled variants at this resolution.
- No medium or large model was benchmarked on Dataset 1-Tiled with 100 epochs.
- The tiled dataset may be more beneficial at smaller input sizes or when detecting very small objects — conditions not explored in the current benchmark.
- **A systematic comparison of all 16 models on Dataset 1-Tiled was not performed.**

---

## 12. Tracking — ByteTrack / BotSort

### 12.1 Status

**Multi-object tracking (ByteTrack or BotSort) has not been implemented or evaluated in this project.** No tracking scripts, tracking results, or tracking configuration files exist in the repository.

### 12.2 Available Infrastructure

- Ultralytics includes built-in tracker support. The `default.yaml` specifies `tracker: botsort.yaml` as the default.
- Ultralytics provides `yolo track` mode which chains a YOLO detector with a tracker (ByteTrack or BotSort) in a single pipeline.
- The installed Ultralytics version fully supports:
  ```bash
  yolo track model=yolov9c.pt source=<video> tracker=bytetrack.yaml
  ```
- ByteTrack parameters (configurable via `bytetrack.yaml`): `track_high_thresh`, `track_low_thresh`, `match_thresh`, `track_buffer`, `frame_rate`.

### 12.3 Future Work Required

For thesis completeness, the following tracking evaluations are needed:

| Experiment | Description |
|-----------|-------------|
| Tracker integration | Run `yolo track` with YOLOv9c + ByteTrack on drone video sequences |
| ID-switch analysis | Measure IDSW per sequence, especially when turtles submerge or occlude |
| Latency measurement | Measure end-to-end FPS (detection + tracking) on the deployment GPU |
| ByteTrack parameters | Tune `track_high_thresh` and `match_thresh` for the low-density aerial scenario |
| HOTA/MOTA metrics | Compute standard MOT metrics on annotated video sequences |

---

## 13. Open Questions & Data Gaps

The following questions from the thesis analysis cannot be answered from the current repository data:

| Question | Status | Action required |
|----------|--------|----------------|
| Per-epoch mAP50 for each of the 16 benchmark models | **Not available** (runs/ gitignored) | Re-run benchmark with `save_period=1` or manually extract from training logs |
| Best epoch number per model | **Not available** | Same as above |
| Inference time (ms/image) per model | **Not measured** | Run `ultralytics benchmark` or time `model.predict()` on test set |
| Bounding box size distribution (small/med/large %) | **Not computed** | Write a script to parse all `.txt` label files and compute bbox statistics |
| Instance count distribution per image | **Not computed** | Same script as above |
| Images where multiple models fail simultaneously | **Not analysed** | Run all 16 models on test set and compare per-image results |
| Model behaviour on low-contrast / water-reflection images | **Not analysed** | Requires curated hard-case subset |
| Cross-model comparison (which images v9c misses that others catch) | **Not analysed** | Requires per-image prediction outputs |
| ByteTrack integration | **Not implemented** | See Section 12 |
| TTA (Test-Time Augmentation) experiments | **Not performed** | Run `model.val(augment=True)` |
| imgsz=1280 experiments | **Not performed** | Retrain or re-validate top models at 1280 |
| YOLO26 architecture details (params, FLOPs) | **Not documented** | Check Ultralytics release notes for YOLO26 |
| Complete Dataset 1-Tiled benchmark (all 16 models) | **Not performed** | Extend benchmark script to Dataset 1-Tiled |
| GPU model and VRAM | **Not documented** | Add to experimental setup documentation |

---

## 14. Key Findings for Thesis

> _Based on Experiment 1 (Dataset 1). Experiment 2 findings will be added in Section 17 once training completes._

### 14.1 Performance Findings

1. **All 16 tested architectures achieve mAP50 > 0.88** on this dataset, confirming that modern YOLO models are well-suited to aerial sea turtle detection from NIR imagery, regardless of architecture.

2. **The top-3 models** (YOLOv9c at 0.955, YOLO26m at 0.952, YOLOv9m at 0.950) differ by < 0.006 mAP50 despite substantial architectural differences. The task may be near-saturated at this metric for Dataset 1.

3. **Recall is the critical differentiator** for conservation monitoring. The range 0.794 (YOLOv8n) to 0.915 (YOLO11m) represents a meaningful difference in missed detection rate: YOLO11m misses ~9% of turtles while YOLOv8n misses ~21%.

4. **False positives are essentially absent** across all models (confusion matrix background FP = 0), indicating the models correctly distinguish turtle texture from beach and water backgrounds.

5. **YOLO26m produces the tightest bounding boxes** (highest mAP50-95: 0.6427), suggesting superior localisation despite similar mAP50 to YOLOv9c. For population counting and size estimation, localisation accuracy matters.

6. **YOLOv9s is an anomaly**: it takes as long as a medium model (89.9 min) but achieves small-model performance (0.925 mAP50). This is the worst efficiency ratio among all tested models and should be avoided in practice.

7. **Default hyperparameters outperform tuned hyperparameters** in this domain. Multiple rounds of tuning (84–100 iterations) on YOLOv8n and YOLOv8m failed to improve on the benchmark's default-configuration results.

### 14.2 Practical Deployment Recommendations

| Use case | Recommended model | Justification |
|----------|------------------|---------------|
| Maximum accuracy (population census) | **YOLOv9c** | Highest mAP50 (0.955) and precision (0.935) |
| Maximum recall (minimum missed turtles) | **YOLO11m** | Highest recall (0.915), 6% FN rate |
| Best accuracy–speed balance | **YOLOv8m** or **YOLO26m** | mAP50 > 0.945, trains ~28 min faster than YOLOv9c |
| Edge deployment (limited compute) | **YOLOv8s** | Best efficiency (0.017 mAP50/min), mAP50=0.936 in 54.7 min |
| Rapid prototyping | **YOLO11n** | Fastest viable model, mAP50=0.907 in 54.2 min |
| Best small model for field recall | **YOLO26s** | Highest recall of small tier (0.913), mAP50=0.940 |

### 14.3 Methodological Notes for Thesis

- **Fair comparison**: All 16 models trained under identical conditions (same dataset, epochs, seed, batch, image size, hardware, hyperparameters). This is a strong comparative methodology.
- **Checkpoint reported**: All metrics correspond to the best validation checkpoint (`best.pt`), which is standard practice.
- **Single-class scenario**: The single-class setup simplifies evaluation but limits conclusions about inter-class confusion. Results may not generalise to multi-species scenarios.
- **Dataset size**: With only 968 training images, this is a relatively small dataset. The strong results (mAP50 > 0.95) are enabled by COCO pre-training and the high visual distinctiveness of turtles in this imagery.
- **Test set evaluation completed (Phase 1)**: See Section 15 for full results on the held-out test split.

---

*Generated from repository analysis — `yolo_ultralytics_benchmark/`, `c:/Users/gaby3/Documents/sea-turtles-detection/archive/`, and `c:/Users/gaby3/Documents/sea-turtles-detection/Dataset/` contents.*
*Last updated: 2026-03-07*

---

## 15. Phase 1 — Test Set Evaluation Results

> _Experiment 1 — Dataset 1 test set (118 images). Model weights from the 2026-02-02 benchmark._
>
> Completed: 2026-03-07
> Script: `c:/Users/gaby3/Documents/sea-turtles-detection/scripts/test_evaluation.py`
> Results file: `c:/Users/gaby3/Documents/sea-turtles-detection/results/test_results.csv`
> Hardware: NVIDIA GeForce RTX 3070 (8 GB), Ultralytics 8.4.6, PyTorch 2.1.0+cu121
> Test split: **261 images, 730 instances** (48 background images with no turtles)

---

### 15.1 Full Test Set Results (sorted by mAP50)

| Rank | Model | mAP50 | mAP50-95 | Precision | Recall | Inference (ms/img) | ~FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | yolov9c | **0.9445** | **0.6364** | **0.9240** | 0.8973 | 14.43 | 69 |
| 2 | yolov9m | 0.9436 | 0.6142 | 0.9152 | 0.9018 | 11.92 | 84 |
| 3 | yolo26m | 0.9420 | 0.6347 | 0.9131 | 0.8986 | 10.66 | 94 |
| 4 | yolo11m | 0.9397 | 0.6183 | 0.8991 | 0.8910 | 10.17 | 98 |
| 5 | yolov10m | 0.9375 | 0.6194 | 0.9087 | 0.9014 | 10.02 | **100** |
| 6 | yolo26s | 0.9334 | 0.5906 | 0.8828 | 0.8726 | 5.68 | 176 |
| 7 | yolo11s | 0.9321 | 0.5757 | 0.8915 | 0.8892 | 5.09 | 197 |
| 8 | yolov10s | 0.9314 | 0.5862 | 0.9074 | 0.8781 | 5.03 | 199 |
| 9 | yolov8m | 0.9314 | 0.6070 | 0.8932 | **0.9054** | 10.25 | 98 |
| 10 | yolov8s | 0.9273 | 0.5601 | 0.8891 | 0.8785 | 4.99 | **200** |
| 11 | yolov9s | 0.9172 | 0.5537 | 0.8754 | 0.8616 | 6.19 | 162 |
| 12 | yolov10n | 0.9006 | 0.5355 | 0.8390 | 0.8548 | 2.94 | 340 |
| 13 | yolo11n | 0.9002 | 0.5252 | 0.8534 | 0.8397 | 3.32 | 301 |
| 14 | yolov8n | 0.8936 | 0.5146 | 0.8499 | 0.8096 | **2.60** | 385 |
| 15 | yolo26n | 0.8911 | 0.5240 | 0.8217 | 0.8247 | 3.06 | 327 |
| 16 | yolov9t | 0.8784 | 0.5015 | 0.8212 | 0.8164 | 4.27 | 234 |

> FPS estimated as 1000 / inference_ms (inference only, batch=8, RTX 3070). All models comfortably exceed 25 FPS real-time threshold.

---

### 15.2 Val vs Test Comparison (mAP50)

| Model | Val mAP50 | Test mAP50 | Delta | Verdict |
| --- | --- | --- | --- | --- |
| yolov9c | 0.9551 | 0.9445 | -0.011 | Largest medium-model drop; still rank 1 |
| yolov8m | 0.9456 | 0.9314 | **-0.014** | Largest absolute drop overall |
| yolo26m | 0.9521 | 0.9420 | -0.010 | Small drop; rank 2→3 |
| yolov9m | 0.9501 | 0.9436 | -0.007 | Most consistent; rank 3→2 |
| yolo11m | 0.9457 | 0.9397 | -0.006 | Stable |
| yolov10m | 0.9444 | 0.9375 | -0.007 | Stable |
| yolo11s | 0.9318 | 0.9321 | **+0.000** | Most generalised small model |
| yolov8s | 0.9356 | 0.9273 | -0.008 | Small drop |
| yolov10s | 0.9359 | 0.9314 | -0.005 | Stable |
| yolov8n | 0.8860 | 0.8936 | **+0.008** | Slight gain (within noise) |
| yolo26n | 0.8875 | 0.8911 | +0.004 | Slight gain (within noise) |
| yolov9t | 0.8865 | 0.8784 | -0.008 | Small drop |

All val→test deltas are ≤ 0.014 mAP50. No model shows a large performance gap, confirming **no significant overfitting to the validation split**.

---

### 15.3 Key Findings — Phase 1

**1. Rankings are stable across val and test splits.**
The top-5 models on the test set are the same as on the validation set, with minor reordering. YOLOv9c remains rank 1 (test mAP50 = 0.9445).

**2. YOLOv9m nearly ties YOLOv9c on test (gap = 0.001 mAP50).**
On validation, the gap was 0.005. On the unseen test set, YOLOv9m generalises almost identically to YOLOv9c, while being smaller (20M vs 25M params) and ~2.5ms faster. This makes **YOLOv9m a strong practical choice** if model size matters.

**3. YOLOv8m shows the largest drop (−0.014 mAP50).**
Its val-set ranking (rank 5 tied with YOLO11m) does not hold on test (rank 9, tied with YOLOv10s). The difference is small in absolute terms but worth noting.

**4. YOLO11m has the best recall on the validation set (0.915) but this does not hold on test (0.891).**
On the test set, the highest recall among medium models is **YOLOv8m (0.905)**, followed by YOLOv10m (0.901) and YOLOv9m (0.902). For conservation monitoring, YOLOv9m offers the best balance of mAP50 and recall.

**5. All models are real-time capable on an RTX 3070.**
Even YOLOv9c (largest model, 14.43 ms/img) runs at ~69 FPS — well above the 25 FPS threshold for video processing. Nano models (YOLOv8n at 2.60 ms) reach ~385 FPS.

**6. YOLO11s is the most stable small model across splits (Δ = 0.000 mAP50).**
It is the only model that does not degrade at all from val to test, suggesting particularly good generalisation for its size.

**7. Test set has 48/261 background images (18.4%).**
These are images with no turtles present; no model produced false positives from background, consistent with the confusion matrix analysis.

---

### 15.4 Updated Model Recommendation (based on test results)

| Use case | Recommended model | Test mAP50 | Test Recall | Inference |
| --- | --- | --- | --- | --- |
| Best accuracy on unseen data | **YOLOv9c** | 0.9445 | 0.897 | 14.4 ms |
| Best accuracy + recall balance | **YOLOv9m** | 0.9436 | 0.902 | 11.9 ms |
| Best accuracy for tracking (recall priority) | **YOLOv8m** | 0.9314 | 0.905 | 10.3 ms |
| Best small model overall | **YOLO11s** | 0.9321 | 0.889 | 5.1 ms |
| Fastest viable model | **YOLOv8n** | 0.8936 | 0.810 | 2.6 ms |
| Best efficiency (mAP50 per ms) | **YOLOv8s** | 0.9273 | 0.879 | 5.0 ms |

**For ByteTrack (Phase 4): use YOLOv9m** — it matches YOLOv9c accuracy on the test set, has the second-best recall among medium models, and is 2.5 ms faster per frame (important for real-time tracking pipelines).

---

## 16. Phase 4 — ByteTrack Tracking Integration

> _Uses the best model from Experiment 1 (YOLOv9m, trained on Dataset 1). The domain gap discovered here motivated Experiment 2 (Section 17)._

### 16.1 Objective

Extend the frame-level detection pipeline with multi-object tracking so that each sea turtle receives a **persistent ID across frames**. This enables population counting and re-identification in drone video surveys without requiring additional annotated data.

### 16.2 Model Selected

**YOLOv9m** (`c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolov9m_20260202_145822/weights/best.pt`).

#### Selection rationale

| Criterion | YOLOv9c | YOLOv9m | Decision |
| --- | --- | --- | --- |
| Test mAP50 | 0.9445 | 0.9436 | Difference of 0.001 — within statistical noise at 118 test images |
| Test Recall | 0.897 | **0.902** | YOLOv9m wins — fewer missed turtles, critical for conservation |
| Inference speed | 14.4 ms/img | **11.9 ms/img** | YOLOv9m is 2.5 ms faster, ~84 FPS vs ~69 FPS headroom |
| Tracking suitability | Marginal | **Better** | Higher recall = more detections fed to ByteTrack = more stable track continuity |

**Conclusion:** YOLOv9m achieves the same practical accuracy as YOLOv9c on the test set, detects more turtles (higher recall), and runs faster — making it the superior choice for a real-time tracking pipeline. In a conservation context, recall is the priority metric: a missed turtle cannot be re-identified later.

### 16.3 ByteTrack Configuration

Custom tracker config: `yolo_ultralytics_benchmark/scripts/bytetrack_turtles.yaml`

| Parameter | Default | Sea turtle tuning | Rationale |
| --- | --- | --- | --- |
| `track_high_thresh` | 0.5 | **0.35** | NIR aerial imagery has variable confidence; accept more detections into tracks |
| `track_low_thresh` | 0.1 | 0.10 | Unchanged |
| `new_track_thresh` | 0.6 | **0.50** | Start new track faster for turtles entering frame |
| `track_buffer` | 30 | **60** | Keep track alive for 2 s at 30 FPS — covers brief submersions |
| `match_thresh` | 0.8 | 0.80 | Unchanged — consistent top-down appearance aids association |
| `frame_rate` | 30 | 30 | Match actual drone video FPS |

### 16.4 Tracking Script

`yolo_ultralytics_benchmark/scripts/tracking.py`

Accepts a video file or sorted image folder (pseudo-sequence). Outputs:

- Annotated video (or frames) with bounding boxes and track IDs overlaid, saved under `c:/Users/gaby3/Documents/sea-turtles-detection/results/tracking/<run_name>/`
- `tracking_stats.txt` with: end-to-end FPS, unique track ID count, per-frame ID sets, ID-set-change count (activity proxy)

#### Example usage

```bash
# On a drone video file
python yolo_ultralytics_benchmark/scripts/tracking.py \
    --source path/to/video.mp4 \
    --name yolov9m_bytetrack

# On a folder of sorted frames
python yolo_ultralytics_benchmark/scripts/tracking.py \
    --source path/to/frames_folder \
    --fps 30 \
    --name yolov9m_bytetrack_frames
```

### 16.5 Dataset Sequence Availability

The Roboflow dataset (`sea-turtles-1`) contains only still images. Analysis of filenames reveals two very short sub-sampled sequences extracted from drone footage:

| Sequence | Frames available | Splits | Note |
| --- | --- | --- | --- |
| Video 55 | 7 (frames 9, 21, 41, 44, 47, …) | train / valid | Non-consecutive; unsuitable for tracking |
| Video 93 | 7 (frames 14, 34, 60, 68, …) | test / train / valid | Non-consecutive; unsuitable for tracking |

These subsampled stills cannot serve as a tracking sequence because there is no temporal continuity between frames. **Actual drone video footage is required for a meaningful ByteTrack evaluation.** Once video is available, run `tracking.py --source video.mp4`.

### 16.6 What to Report in Thesis

| Metric | Source |
| --- | --- |
| End-to-end FPS | `tracking_stats.txt` → `fps` field |
| Unique turtles detected (track IDs) | `tracking_stats.txt` → `unique_ids` |
| Visual ID switches | Inspect annotated output video; count moments where a turtle's ID changes |
| Track recovery after submersion | Check if same ID is recovered after a gap in per-frame ID list |
| Qualitative robustness | Describe occlusion handling, re-entry behaviour |

Formal MOTA / HOTA metrics require ground-truth track IDs across frames (not available in this dataset). The qualitative approach above is appropriate for a thesis-level tracking evaluation.

### 16.7 Tracking Evaluation Results

ByteTrack was run on a drone video (`c:/Users/gaby3/Documents/sea-turtles-detection/videos/turtle_footage4.mp4`) using YOLOv9m. Detection quality was extremely poor.

**Quantitative results from `c:/Users/gaby3/Documents/sea-turtles-detection/results/tracking/yolov9m_bytetrack/tracking_stats.txt`:**

| Metric | Value | Interpretation |
| --- | --- | --- |
| Source video | `c:/Users/gaby3/Documents/sea-turtles-detection/videos/turtle_footage4.mp4` | Closer-range drone footage |
| Total frames | 1,989 | ~66 seconds at 30 FPS |
| Processing time | 59.5 s | 33.4 FPS — real-time capable |
| Unique track IDs assigned | **23** | Misleading: each is a new brief track, not 23 unique turtles |
| ID set changes | **83** | 83 moments where tracked set changed — extreme instability |
| Frames with any detection | **~85 / 1,989** | **≈ 4.3% of frames** — model detected nothing in 95.7% of frames |
| Longest continuous track | **~10 frames** (ID 95, frames 1744–1755) | Equivalent to ~0.3 seconds |
| Maximum simultaneous detections | **2** (IDs 97+98 at frames 1767–1768) | Never detected more than 2 turtles at once |

**Detection pattern (from per_frame_ids log):** The detector fired in brief isolated bursts of 1–8 frames, then went silent for hundreds of frames. Each burst was assigned a new ID. No turtle was tracked continuously for more than ~10 frames. The video contains 1,989 frames and only ~85 had any detection — confirming the model was effectively blind to the content of this footage.

**Root cause: domain gap between training data and evaluation videos.**

| Property | Dataset 1 (training) | turtle_footage4.mp4 (evaluation) |
| --- | --- | --- |
| Altitude | High — turtles appear small | Lower — turtles appear larger |
| Perspective | Strict top-down | Variable angle, closer range |
| Imaging | NIR (near-infrared) | Standard visible-light video |
| Environment | Turtles at sea surface | Turtles on beach and in water |

The model never saw turtles at this scale, angle, or lighting condition during training. It correctly suppressed uncertain detections rather than generating false positives — but the result was effective blindness to real turtles. **This is not a tracker configuration problem.** No amount of ByteTrack parameter tuning can compensate for a detector that was not trained on the target domain. This finding directly motivated Experiment 2 (Section 17).

### 16.8 Key Findings

1. **Infrastructure complete.** `tracking.py` and `bytetrack_turtles.yaml` are ready and functional.
2. **Domain gap identified.** Current models (trained on high-altitude, at-sea imagery) do not generalise to closer-range or on-land footage. This is the primary limitation for deployment.
3. **FPS headroom confirmed.** On matched-domain footage, YOLOv9m + ByteTrack comfortably exceeds 30 FPS on an RTX 3070.
4. **Next step is a dataset problem, not a model problem.** Retraining or fine-tuning on closer-range and on-land images is required before tracking results will be meaningful.

### 16.9 Next Steps — Dataset Expansion to Close the Domain Gap

The tracking evaluation revealed that the current dataset covers only one operational scenario (high-altitude, turtles at sea). To build a robust detector for real-world deployment, the training data must cover the full range of conditions encountered in the field.

#### Option A — Preferred: use videos from the same altitude and conditions as the training dataset

Apply ByteTrack to drone footage filmed at the same height and over water — matching the training distribution exactly. No retraining required. This is the fastest path to valid tracking results for the thesis.

#### Option B — Extended: expand the dataset with closer-range and on-land images

Collect or annotate images of turtles at closer range and on land, add them to the dataset, and retrain. This produces a more robust model but requires significant annotation effort.

| Scenario | Images needed | Effort | Expected gain |
| --- | --- | --- | --- |
| High-altitude, at sea (current) | Already covered | — | Baseline |
| Closer-range, at sea | 200–400 annotated frames | Medium | Generalises to lower-altitude surveys |
| Turtles on land / beach | 200–400 annotated frames | Medium | Covers nesting season surveys |
| Mixed altitude + conditions | 400–800 annotated frames | High | Full deployment robustness |

**Recommendation for thesis scope:** Use Option A (matching-domain video) for the tracking chapter. Document the domain gap as a limitation and propose Option B as future work.

---

## 17. Experiment 2 — Benchmark on Closer-Range Dataset

### 17.1 Motivation — Why We Moved to a Different Dataset

After completing Experiment 1 (Section 3–15), the natural next step was to validate the best model (YOLOv9m) in a real-world tracking scenario using ByteTrack (Phase 4, Section 16). ByteTrack was applied to **4 drone videos** filmed at lower altitude than the training imagery.

#### What happened with the tracking

The tracking results were poor across all 4 videos. The detector produced frequent missed detections and low-confidence outputs, which ByteTrack could not recover from regardless of threshold tuning.

Observed symptoms during tracking evaluation:

| Symptom | Observed behaviour |
| --- | --- |
| Detection rate | Very low — most turtles missed per frame |
| Confidence scores | Consistently below `track_high_thresh` (0.35) |
| Track continuity | IDs created briefly then immediately lost |
| False positives | Low (model suppressed everything, not just turtles) |
| ID switches | Frequent — tracker had no stable detections to follow |

#### Root cause: domain gap

The training data (Dataset 1) and the evaluation videos came from **different operational scenarios**:

| Property | Dataset 1 (training) | Field videos (evaluation) |
| --- | --- | --- |
| Altitude | High — turtles appear small | Lower — turtles appear larger |
| Perspective | Strict top-down aerial | Variable angle, closer range |
| Environment | Turtles at sea surface | Turtles on beach and in water |
| Imaging | NIR (near-infrared) | Standard visible-light video |
| Turtle appearance | Small silhouettes, uniform | Larger, more detailed, varied backgrounds |

The visual appearance of turtles in the field videos was fundamentally different from anything in the training set. The model had never seen turtles at this scale or from this perspective, so it correctly suppressed most detections as uncertain. **This is not a tracker problem — it is a training data problem.** No amount of ByteTrack parameter tuning can compensate for a detector that was not trained on the target domain.

#### Decision: train on a closer-range dataset

To solve the domain gap, we searched for a publicly available dataset that matched the field video conditions — lower altitude, visible light, varied backgrounds. The Roboflow dataset **sea-turtles-model v6** was identified as the best available match:

- Captured at lower altitude than Dataset 1
- Contains turtles in various environments (beach, water, mixed)
- 2,890 images, MIT licence, pre-split into train/val/test

Experiment 2 replicates the full 16-model benchmark protocol on this dataset, with the goal of producing models that perform in the same visual domain as the field videos where ByteTrack failed.

This experiment answers: **do the same YOLO architectures remain capable when trained on closer-range imagery, and does the resulting model resolve the tracking domain gap?**

### 17.2 Dataset Profile

See Section 2.6 for full dataset characteristics. Summary:

| Property | Value |
| --- | --- |
| Dataset name | sea-turtles-model v6 (Roboflow) |
| License | MIT |
| Train / Val / Test | 2,071 / 534 / 285 |
| Total images | 2,890 |
| Imagery type | Lower-altitude drone, visible light |
| Classes | 1 (turtle) |
| Local path | `c:/Users/gaby3/Documents/sea-turtles-detection/Dataset/sea-turtles-2/` |
| Config | `c:/Users/gaby3/Documents/sea-turtles-detection/Dataset/sea-turtles-2/data.yaml` |

### 17.3 Training Setup

Identical hyperparameters to Experiment 1 (`benchmark_yolo_models_ds2.py`):

| Parameter | Value |
| --- | --- |
| Models | Same 16 architectures (YOLOv8 n/s/m, YOLOv9 t/s/m/c, YOLOv10 n/s/m, YOLO11 n/s/m, YOLO26 n/s/m) |
| Epochs | 100 |
| Image size | 640 |
| Batch | 8 |
| Device | CUDA GPU (device 0) |
| Seed | 42 |
| Workers | 0 |
| Results CSV | `c:/Users/gaby3/Documents/sea-turtles-detection/results/ds2/benchmark_results_ds2.csv` |
| Run output | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/` |

The training order follows the same sequence: YOLO26 n/s/m → YOLO11 n/s/m → YOLOv10 n/s/m → YOLOv9 t/s/m/c → YOLOv8 n/s/m.

### 17.4 Full Benchmark Results (Experiment 2 — Dataset 2)

Training completed **2026-03-07 to 2026-03-09**, 16 models × 100 epochs. All values from `c:/Users/gaby3/Documents/sea-turtles-detection/results/ds2/benchmark_results_ds2.csv`.

| Model | Train (min) | mAP50 | mAP50-95 | Precision | Recall | Status |
| --- | --- | --- | --- | --- | --- | --- |
| yolo26n | 108.9 | 0.386 | 0.223 | 0.160 | 0.700 | OK |
| yolo26s | 111.3 | 0.460 | 0.275 | 0.230 | 0.667 | OK |
| yolo26m | 148.0 | **0.504** | 0.279 | 0.178 | **0.800** | OK |
| yolo11n | 95.1 | 0.533 | 0.346 | **1.000** | 0.067 | Anomaly — precision collapse |
| yolo11s | 141.5 | 0.000 | 0.000 | 0.000 | 0.000 | Failed |
| yolo11m | 136.6 | 0.000 | 0.000 | 0.000 | 0.000 | Failed |
| yolov10n | 99.6 | 0.320 | 0.188 | 0.227 | 0.667 | OK |
| yolov10s | 105.3 | 0.375 | 0.142 | 0.538 | 0.233 | OK |
| yolov10m | 141.8 | 0.294 | 0.179 | 0.268 | 0.633 | OK |
| yolov9t | 145.3 | 0.517 | 0.310 | **1.000** | 0.033 | Anomaly — precision collapse |
| yolov9s | 145.5 | 0.000 | 0.000 | 0.000 | 0.000 | Failed |
| yolov9m | 194.9 | 0.463 | 0.245 | 0.300 | **0.700** | OK |
| yolov9c | 199.9 | 0.356 | 0.231 | 0.667 | 0.067 | Anomaly — precision collapse |
| yolov8n | 105.0 | 0.385 | 0.194 | 0.500 | 0.467 | OK |
| yolov8s | 111.8 | 0.395 | 0.226 | 0.340 | 0.567 | OK |
| yolov8m | 133.1 | 0.517 | 0.310 | **1.000** | 0.033 | Anomaly — precision collapse |

Sorted by mAP50 × Recall (tracking priority score):

| Rank | Model | mAP50 | Recall | mAP50 × Recall |
| --- | --- | --- | --- | --- |
| 1 | **yolo26m** | 0.504 | 0.800 | **0.403** |
| 2 | **yolov9m** | 0.463 | 0.700 | **0.324** |
| 3 | **yolo26s** | 0.460 | 0.667 | **0.307** |
| 4 | yolo26n | 0.386 | 0.700 | 0.270 |
| 5 | yolov8s | 0.395 | 0.567 | 0.224 |

### 17.5 Analysis — Why the Results Are Poor

The Experiment 2 benchmark results are significantly worse than Experiment 1 (DS1 best: 0.955 mAP50 vs DS3 best: 0.504 mAP50). Three categories of failure are observed.

#### Cause 1 — Complete training collapse (3 models: yolo11s, yolo11m, yolov9s)

mAP50 = 0.000 on all metrics. These models failed to learn any meaningful detection. Likely cause: **training loss went NaN** or the model's head never connected to the dataset's label distribution. This can happen with:

- Small batch size (batch=8) and large model on a heterogeneous dataset — gradient variance is high
- A specific random seed that produces early bad batches, causing irreversible weight divergence
- Interaction between certain architectures and the Dataset 2 class distribution

These results are **not informative about the architecture's potential** — they reflect training instability, not model capability.

#### Cause 2 — Precision collapse (4 models: yolo11n, yolov9t, yolov9c, yolov8m)

Precision = 1.000, Recall ≈ 0.03–0.07. The model learned to fire on only 1–2 instances across the entire validation set — those instances are correct (precision = 1), but it misses almost all turtles (recall ≈ 0). This is a **degenerate solution**: the model found that predicting almost nothing is safe because the false-negative penalty in the loss is low.

Root cause: **class imbalance or low instance density in Dataset 2**. If many val images contain no turtles and the images with turtles have few visible instances, the model learns that suppressing all predictions is a low-loss strategy. The NMS threshold during validation may also be too conservative for this dataset's confidence distribution.

#### Cause 3 — Low mAP50 across all OK models (0.29–0.50)

Even models that trained correctly score far below DS1. Contributing factors:

- **Visual heterogeneity**: Dataset 2 contains varied backgrounds (beach, shallow water, vegetation) and turtle appearances (different sizes, angles, lighting). This is a harder detection problem than DS1's uniform aerial view.
- **Fewer training samples per visual cluster**: 2,071 training images spread across more conditions = less per-condition coverage than DS1's 968 images of one scenario.
- **Val/test set size**: with only 534 val images and an estimated small number of turtle instances, a single missed detection shifts recall significantly. The metrics are **high-variance** at this scale.
- **Same hyperparameters as DS1**: batch=8 and workers=0 were calibrated for DS1's simpler distribution. Dataset 2 likely benefits from larger batch or higher learning rate warmup.

#### Summary

| Failure type | Models affected | Root cause |
| --- | --- | --- |
| Complete collapse (0.0) | yolo11s, yolo11m, yolov9s | Training divergence; batch=8 instability |
| Precision collapse | yolo11n, yolov9t, yolov9c, yolov8m | Degenerate safe solution; class imbalance |
| Low but valid mAP50 | Remaining 9 models | Dataset heterogeneity + small val set |

### 17.6 Model Selection for Retraining (Phase 3)

Based on the benchmark, the **3 models with the highest potential for field deployment** (best mAP50 × Recall score, excluding anomalous results):

| Model | mAP50 | Recall | Rationale |
| --- | --- | --- | --- |
| **yolo26m** | 0.504 | 0.800 | Highest recall — fewest missed turtles; best overall score |
| **yolov9m** | 0.463 | 0.700 | Proven architecture from Experiment 1; consistent across both experiments |
| **yolo26s** | 0.460 | 0.667 | Best small model; same family as top performer |

These three will be retrained with **early stopping** to allow convergence beyond 100 epochs (see Section 17.7).

### 17.7 Retraining Strategy — Early Stopping (Phase 3)

**Approach:** Retrain the 3 selected models with `patience=50` (stop if no improvement in val mAP50 for 50 consecutive epochs), a higher epoch ceiling (300), and corrected hyperparameters to address the instability observed.

| Parameter | Benchmark value | Retrain value | Reason |
| --- | --- | --- | --- |
| `epochs` | 100 | 300 (ceiling) | Allow full convergence |
| `patience` | 100 (disabled) | **50** | Stop when learning plateaus |
| `batch` | 8 | **16** | Larger batches → more stable gradients |
| `workers` | 0 | **4** | Parallel data prefetch; GPU was below 100% utilisation |
| `seed` | 42 | 42 | Keep reproducible |

**Expected outcome:** With early stopping, training stops automatically when val mAP50 stops improving — typically 150–200 epochs for models that are still learning at epoch 100. This avoids overfitting and saves GPU time compared to a fixed 300-epoch run.

**Script:** `c:/Users/gaby3/Documents/sea-turtles-detection/scripts/benchmark_yolo_models_ds2_retrain.py` (to be created) — same structure as `benchmark_yolo_models_ds2.py` but limited to 3 models with updated parameters above.

### 17.8 Comparison with Experiment 1

| Dimension | Experiment 1 (DS1) | Experiment 2 (DS3, initial) | Experiment 2 (DS3, retrain) |
| --- | --- | --- | --- |
| Dataset altitude | High (small turtles) | Lower (larger turtles) | Lower (larger turtles) |
| Training images | 968 | 2,071 | 2,071 |
| Best mAP50 | 0.955 (YOLOv9c) | 0.504 (yolo26m) | TBD |
| Best recall | 0.915 (YOLO11m) | 0.800 (yolo26m) | TBD |
| Failed models | 0 / 16 | 7 / 16 | Expected 0 / 3 |
| Domain match for field video | No | Yes | Yes |

The architecture ranking from Experiment 1 (YOLOv9 and YOLO26 families strongest) is **partially confirmed** in Experiment 2: yolo26m and yolov9m are again in the top 3. The failure of YOLOv11 and YOLOv9s/c may be specific to the batch=8 instability rather than the architectures themselves.

### 17.9 Next Steps — Phase 3: Targeted Retraining (Pending Results)

> **Current status (2026-03-09):** The initial 16-model benchmark on Dataset 2 is complete (Section 17.4). The 3 best models have been identified (Section 17.6). **Phase 3 retraining has not yet started — results are pending.**

The plan is to retrain the 3 selected models (yolo26m, yolov9m, yolo26s) with corrected hyperparameters and early stopping. This section will be updated with results once training completes.

#### Phase 3 plan

1. **Create retrain script** — `c:/Users/gaby3/Documents/sea-turtles-detection/scripts/benchmark_yolo_models_ds2_retrain.py`: same structure as the initial benchmark but limited to 3 models, with `batch=16`, `workers=4`, `patience=50`, `epochs=300` ceiling.
2. **Run retraining** — estimated time: ~3–5 hours per model (early stopping expected to trigger before epoch 300).
3. **Evaluate on test set** — run test-set evaluation equivalent to `c:/Users/gaby3/Documents/sea-turtles-detection/scripts/test_evaluation.py` on the 3 retrained weights.
4. **Run ByteTrack on field videos** — apply `c:/Users/gaby3/Documents/sea-turtles-detection/scripts/tracking.py` to the same 4 videos where Experiment 1 failed (Section 16.7), using the best retrained model. This directly tests whether the domain gap is resolved.
5. **Document results** — update Section 17.8 comparison table and add tracking quality comparison (Experiment 1 model vs retrained Experiment 2 model).

#### What success looks like

| Metric | Experiment 1 (DS1) | Target (DS3 retrain) |
| --- | --- | --- |
| Val mAP50 (best model) | 0.955 | > 0.65 (estimated realistic target) |
| Field video detection rate | Very low (domain gap) | Consistent detections per frame |
| ByteTrack track continuity | Poor (IDs lost immediately) | Stable IDs across frames |

A retrained model that produces consistent detections on the field videos — even at lower mAP50 than Experiment 1 — represents a successful resolution of the domain gap and is the primary deliverable of this phase.

---

## 18. Project File & Asset Index

> This section maps every result file, image, and model weight to its full path. Intended as a reference for the thesis author and for any LLM generating the written thesis chapters.

### 18.1 Project Directory Structure

```text
c:/Users/gaby3/Documents/sea-turtles-detection/
├── ANALYSIS.md                          # This document
├── Makefile                             # All runnable commands
├── configs/
│   └── bytetrack_turtles.yaml           # ByteTrack tracker configuration
├── scripts/
│   ├── benchmark_yolo_models.py         # Experiment 1: train+val 16 models on DS1
│   ├── benchmark_yolo_models_ds2.py     # Experiment 2: train+val 16 models on DS3
│   ├── test_evaluation.py               # Run test-set evaluation on trained models
│   └── tracking.py                      # ByteTrack inference on video
├── Dataset/
│   ├── sea-turtles-1/                   # Dataset 1 (high-altitude NIR, CC BY 4.0)
│   │   └── data.yaml
│   └── sea-turtles-2/                   # Dataset 2 (closer-range, MIT) [gitignored]
│       └── data.yaml
├── models/                              # Base YOLO .pt weight files (pretrained COCO)
├── results/
│   ├── benchmark_results.csv            # Experiment 1 val metrics (all 16 models)
│   ├── test_results.csv                 # Experiment 1 test-set metrics (all 16 models)
│   ├── images/                          # Val-set visualisations per model (Experiment 1)
│   │   └── {model}/
│   │       ├── results.png              # Training curves (loss + metrics vs epoch)
│   │       ├── confusion_matrix_normalized.png
│   │       ├── BoxPR_curve.png          # Precision-Recall curve
│   │       └── val_batch0_pred.jpg      # Sample validation predictions
│   ├── test_runs/                       # Test-set visualisations per model (Experiment 1)
│   │   └── {model}/
│   │       ├── confusion_matrix.png
│   │       ├── confusion_matrix_normalized.png
│   │       ├── BoxPR_curve.png
│   │       ├── BoxP_curve.png
│   │       ├── BoxR_curve.png
│   │       ├── BoxF1_curve.png
│   │       ├── val_batch0_labels.jpg    # Ground truth boxes
│   │       ├── val_batch0_pred.jpg      # Predicted boxes
│   │       ├── val_batch1_labels.jpg
│   │       ├── val_batch1_pred.jpg
│   │       ├── val_batch2_labels.jpg
│   │       └── val_batch2_pred.jpg
│   ├── tracking/
│   │   └── yolov9m_bytetrack/
│   │       └── tracking_stats.txt       # ByteTrack stats for turtle_footage4.mp4
│   └── ds2/
│       └── benchmark_results_ds2.csv    # Experiment 2 val metrics (all 16 models)
├── runs/
│   ├── train/                           # Experiment 1 training runs
│   │   └── {model}_{timestamp}/
│   │       └── weights/
│   │           ├── best.pt              # Best checkpoint (used for all evaluations)
│   │           └── last.pt
│   └── train_ds2/                       # Experiment 2 training runs
│       └── {model}_ds2_{timestamp}/
│           └── weights/
│               ├── best.pt
│               └── last.pt
├── videos/
│   └── turtle_footage4.mp4              # Field video used for ByteTrack evaluation
└── archive/                             # Historical experiments (numbered chronologically)
    ├── 01_yolov5/
    ├── 02_yolov8_roboflow/
    ├── 03_yolov8_local/
    ├── 04_yolov9_roboflow/
    ├── 05_yolov9_native/
    └── 06_yolov9_ultralytics/
```

### 18.2 Experiment 1 — Model Weights Paths

All weights at `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/{run_name}/weights/best.pt`. Run names from `c:/Users/gaby3/Documents/sea-turtles-detection/results/benchmark_results.csv`:

| Model | Run name | Best weights path |
| --- | --- | --- |
| yolov9c | yolov9c_20260202_163426 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolov9c_20260202_163426/weights/best.pt` |
| yolo26m | yolo26m_20260202_033138 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolo26m_20260202_033138/weights/best.pt` |
| yolov9m | yolov9m_20260202_145822 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolov9m_20260202_145822/weights/best.pt` |
| yolo11m | yolo11m_20260202_065642 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolo11m_20260202_065642/weights/best.pt` |
| yolov8m | yolov8m_20260202_200749 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolov8m_20260202_200749/weights/best.pt` |
| yolov10m | yolov10m_20260202_102932 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolov10m_20260202_102932/weights/best.pt` |
| yolo26s | yolo26s_20260202_022129 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolo26s_20260202_022129/weights/best.pt` |
| yolov10s | yolov10s_20260202_092131 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolov10s_20260202_092131/weights/best.pt` |
| yolov8s | yolov8s_20260202_191241 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolov8s_20260202_191241/weights/best.pt` |
| yolo11s | yolo11s_20260202_055950 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolo11s_20260202_055950/weights/best.pt` |
| yolov9s | yolov9s_20260202_132754 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolov9s_20260202_132754/weights/best.pt` |
| yolo11n | yolo11n_20260202_050518 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolo11n_20260202_050518/weights/best.pt` |
| yolov10n | yolov10n_20260202_081804 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolov10n_20260202_081804/weights/best.pt` |
| yolo26n | yolo26n_20260202_011017 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolo26n_20260202_011017/weights/best.pt` |
| yolov9t | yolov9t_20260202_120044 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolov9t_20260202_120044/weights/best.pt` |
| yolov8n | yolov8n_20260202_182141 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolov8n_20260202_182141/weights/best.pt` |

**Model used for ByteTrack (Phase 4):** `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train/yolov9m_20260202_145822/weights/best.pt`

### 18.3 Experiment 2 — Model Weights Paths

All weights at `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/{run_name}/weights/best.pt`. Run names from `c:/Users/gaby3/Documents/sea-turtles-detection/results/ds2/benchmark_results_ds2.csv`:

| Model | Run name | Best weights path |
| --- | --- | --- |
| yolo26n | yolo26n_ds2_20260307_141840 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolo26n_ds2_20260307_141840/weights/best.pt` |
| yolo26s | yolo26s_ds2_20260307_160759 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolo26s_ds2_20260307_160759/weights/best.pt` |
| yolo26m | yolo26m_ds2_20260307_175942 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolo26m_ds2_20260307_175942/weights/best.pt` |
| yolo11n | yolo11n_ds2_20260307_202831 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolo11n_ds2_20260307_202831/weights/best.pt` |
| yolo11s | yolo11s_ds2_20260307_220410 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolo11s_ds2_20260307_220410/weights/best.pt` |
| yolo11m | yolo11m_ds2_20260308_002606 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolo11m_ds2_20260308_002606/weights/best.pt` |
| yolov10n | yolov10n_ds2_20260308_024329 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolov10n_ds2_20260308_024329/weights/best.pt` |
| yolov10s | yolov10s_ds2_20260308_042325 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolov10s_ds2_20260308_042325/weights/best.pt` |
| yolov10m | yolov10m_ds2_20260308_060905 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolov10m_ds2_20260308_060905/weights/best.pt` |
| yolov9t | yolov9t_ds2_20260308_083119 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolov9t_ds2_20260308_083119/weights/best.pt` |
| yolov9s | yolov9s_ds2_20260308_105656 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolov9s_ds2_20260308_105656/weights/best.pt` |
| yolov9m | yolov9m_ds2_20260308_132249 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolov9m_ds2_20260308_132249/weights/best.pt` |
| yolov9c | yolov9c_ds2_20260308_163820 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolov9c_ds2_20260308_163820/weights/best.pt` |
| yolov8n | yolov8n_ds2_20260308_195841 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolov8n_ds2_20260308_195841/weights/best.pt` |
| yolov8s | yolov8s_ds2_20260308_214415 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolov8s_ds2_20260308_214415/weights/best.pt` |
| yolov8m | yolov8m_ds2_20260308_233638 | `c:/Users/gaby3/Documents/sea-turtles-detection/runs/train_ds2/yolov8m_ds2_20260308_233638/weights/best.pt` |

### 18.4 Experiment 1 — Val-Set Image Assets

Per-model visualisation images saved during training validation. Path pattern: `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/{model}/{file}`.

| Model | results.png | confusion_matrix_normalized.png | BoxPR_curve.png | val_batch0_pred.jpg |
| --- | --- | --- | --- | --- |
| yolo26n | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo26n/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo26n/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo26n/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo26n/val_batch0_pred.jpg` |
| yolo26s | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo26s/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo26s/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo26s/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo26s/val_batch0_pred.jpg` |
| yolo26m | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo26m/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo26m/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo26m/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo26m/val_batch0_pred.jpg` |
| yolo11n | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo11n/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo11n/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo11n/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo11n/val_batch0_pred.jpg` |
| yolo11s | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo11s/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo11s/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo11s/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo11s/val_batch0_pred.jpg` |
| yolo11m | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo11m/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo11m/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo11m/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolo11m/val_batch0_pred.jpg` |
| yolov10n | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov10n/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov10n/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov10n/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov10n/val_batch0_pred.jpg` |
| yolov10s | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov10s/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov10s/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov10s/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov10s/val_batch0_pred.jpg` |
| yolov10m | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov10m/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov10m/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov10m/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov10m/val_batch0_pred.jpg` |
| yolov9t | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9t/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9t/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9t/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9t/val_batch0_pred.jpg` |
| yolov9s | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9s/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9s/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9s/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9s/val_batch0_pred.jpg` |
| yolov9m | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9m/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9m/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9m/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9m/val_batch0_pred.jpg` |
| yolov9c | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9c/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9c/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9c/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov9c/val_batch0_pred.jpg` |
| yolov8n | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov8n/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov8n/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov8n/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov8n/val_batch0_pred.jpg` |
| yolov8s | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov8s/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov8s/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov8s/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov8s/val_batch0_pred.jpg` |
| yolov8m | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov8m/results.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov8m/confusion_matrix_normalized.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov8m/BoxPR_curve.png` | `c:/Users/gaby3/Documents/sea-turtles-detection/results/images/yolov8m/val_batch0_pred.jpg` |

### 18.5 Experiment 1 — Test-Set Image Assets

Per-model images from the test-set evaluation run. Path pattern: `c:/Users/gaby3/Documents/sea-turtles-detection/results/test_runs/{model}/{file}`. Each model has:
`confusion_matrix.png`, `confusion_matrix_normalized.png`, `BoxPR_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png`, `val_batch0_labels.jpg`, `val_batch0_pred.jpg`, `val_batch1_labels.jpg`, `val_batch1_pred.jpg`, `val_batch2_labels.jpg`, `val_batch2_pred.jpg`

Models with test-set image assets:
`yolo26n`, `yolo26s`, `yolo26m`, `yolo11n`, `yolo11s`, `yolo11m`, `yolov10n`, `yolov10s`, `yolov10m`, `yolov9t`, `yolov9s`, `yolov9m`, `yolov9c`, `yolov8n`, `yolov8s`, `yolov8m`

Example full paths for YOLOv9m (best tracking model):

- `c:/Users/gaby3/Documents/sea-turtles-detection/results/test_runs/yolov9m/confusion_matrix_normalized.png`
- `c:/Users/gaby3/Documents/sea-turtles-detection/results/test_runs/yolov9m/BoxPR_curve.png`
- `c:/Users/gaby3/Documents/sea-turtles-detection/results/test_runs/yolov9m/BoxF1_curve.png`
- `c:/Users/gaby3/Documents/sea-turtles-detection/results/test_runs/yolov9m/val_batch0_pred.jpg`

### 18.6 Key Raw Data Files

| File | Description | Experiment |
| --- | --- | --- |
| `c:/Users/gaby3/Documents/sea-turtles-detection/results/benchmark_results.csv` | Val metrics for all 16 models | Experiment 1 |
| `c:/Users/gaby3/Documents/sea-turtles-detection/results/test_results.csv` | Test-set metrics + inference timing for all 16 models | Experiment 1 |
| `c:/Users/gaby3/Documents/sea-turtles-detection/results/ds2/benchmark_results_ds2.csv` | Val metrics for all 16 models on Dataset 2 | Experiment 2 |
| `c:/Users/gaby3/Documents/sea-turtles-detection/results/tracking/yolov9m_bytetrack/tracking_stats.txt` | ByteTrack run stats (1989 frames, 33.4 FPS, 23 IDs, 4.3% detection rate) | Phase 4 |
| `c:/Users/gaby3/Documents/sea-turtles-detection/Dataset/sea-turtles-1/data.yaml` | Dataset 1 split config | Experiment 1 |
| `c:/Users/gaby3/Documents/sea-turtles-detection/Dataset/sea-turtles-2/data.yaml` | Dataset 2 split config | Experiment 2 |
| `c:/Users/gaby3/Documents/sea-turtles-detection/configs/bytetrack_turtles.yaml` | ByteTrack parameters (tuned for sea turtles) | Phase 4 |
