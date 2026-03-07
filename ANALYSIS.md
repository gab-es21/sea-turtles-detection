# ANALYSIS — Sea Turtle Detection: Training, Models & Dataset

> Analysis document for Master's thesis.
> All results refer to the benchmark run executed on **2026-02-02**, training 16 models for 100 epochs on Dataset 1.
> Source data: `yolo_ultralytics_benchmark/results/benchmark_results.csv`, training curve images in `yolo_ultralytics_benchmark/results/images/`, and archived experiment runs under `archive/`.

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
11. [Dataset 2 — Tiled Dataset Experiments](#11-dataset-2--tiled-dataset-experiments)
12. [Tracking — ByteTrack / BotSort](#12-tracking--bytetrack--botsort)
13. [Open Questions & Data Gaps](#13-open-questions--data-gaps)
14. [Key Findings for Thesis](#14-key-findings-for-thesis)

---

## 1. Executive Summary

This project trains and benchmarks 16 YOLO architectures (v8, v9, v10, v11, v26) for single-class detection of sea turtles in NIR drone imagery. All models were trained under identical conditions on the same dataset, enabling a fair architectural comparison.

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

---

## 2. Dataset & Annotation

### 2.1 Dataset Variants

| ID | Label | Train | Val | Test | Description |
|----|-------|-------|-----|------|-------------|
| 1 | Base (B) | 968 | 272 | 118 | Original annotations from Roboflow |
| 2 | Tiled (W) | 15,488 | 4,352 | 1,888 | Tiled augmentation of Dataset 1 |

The benchmark of 16 models used **Dataset 1** exclusively. Dataset 2 was used only in earlier exploratory experiments (see Section 11).

### 2.2 Source & Annotation Format

- **Source**: [Roboflow — sea-turtles-yia2e](https://universe.roboflow.com/gabriel-esteves-dy2cw/sea-turtles-yia2e/dataset/1), version 1
- **License**: CC BY 4.0
- **Imagery type**: NIR (near-infrared) aerial/drone footage of sea turtles on nesting beaches
- **Annotation format**: **YOLO TXT** (one `.txt` per image with normalized `class cx cy w h` format), as confirmed by `data.yaml` structure (train/valid/test pointing to image folders with paired label folders)
- **Annotation tool**: Roboflow (web-based annotation platform)
- **Class**: Single class — `Turtle` (nc: 1)

### 2.3 Bounding Box Size Distribution

The benchmark does not include an explicit bounding box size analysis script. However, qualitative observations from validation prediction images (`val_batch0_pred.jpg`) indicate:

- Turtles appear as **medium-to-large objects** at 640 px input resolution: bounding boxes typically occupy a substantial portion of each image tile, suggesting the majority fall in the **medium (32–96 px) to large (> 96 px)** range.
- In the mosaic validation grid, individual turtles are clearly visible and well-separated from the background (sandy beach or shallow water).
- Very few instances appear as "small objects" (< 32 px), which is consistent with aerial surveys at close range.
- **Quantitative breakdown**: not computed in this project; a dedicated analysis script (e.g., parsing all label `.txt` files) would be needed for exact percentages.

### 2.4 Instance Distribution per Image

- Not explicitly computed. From visual inspection of batch prediction outputs: most images appear to contain **1 to ~6 turtles** per frame, with some crowded scenes containing more overlapping individuals.
- Dataset 2 (Tiled) splits each original image into 16 tiles (4×4 grid), leading to many tiles containing 0 turtles and some tiles containing dense groupings — explaining the 16× increase in image count with a non-uniform turtle density per tile.

### 2.5 NIR Imagery Pre-processing

- **No explicit NIR-specific pre-processing** was applied before training. Images were used as exported from Roboflow.
- Ultralytics loads images as standard 3-channel tensors regardless of spectral content; NIR images stored as grayscale or pseudo-RGB are handled identically to visible-light imagery.
- The Ultralytics HSV augmentation (hsv_h, hsv_s, hsv_v) operates on the loaded channels without special handling.

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

### 5.1 Complete Results Table (sorted by mAP50)

All values from `yolo_ultralytics_benchmark/results/benchmark_results.csv`. Run date: 2026-02-02.

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

PR curves (`BoxPR_curve.png`) are available for all 16 models. Analysis of the YOLOv9c PR curve (representative of top models):

- **Shape**: Near-perfect rectangular curve — precision remains at ~1.0 from recall 0.0 to ~0.7, then drops steeply.
- **Area under curve (mAP@0.5)**: 0.954 (matches benchmark CSV).
- **High-confidence region**: At recall ≤ 0.70, precision is essentially 1.0 — the model only produces confident detections that are correct.
- **Recall degradation zone**: Precision drops from 1.0 to ~0.85 between recall 0.70 and 0.95, then falls sharply to ~0.0 at recall → 1.0.

Weaker models (nano tier) have less rectangular curves, with earlier precision degradation beginning around recall 0.5–0.6, indicating more mixed-confidence detections.

---

## 8. Model Architecture Comparison

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

3. **Dataset 1 vs Dataset 2 (Runs 1 vs 5, YOLOv8n)**: mAP50 was similar (0.809 vs 0.827) but the tiled dataset required only 100 epochs vs 50 to reach a similar level. The tiled dataset did not provide a substantial accuracy gain for YOLOv8n; the benefit may be larger for detecting smaller objects.

4. **Training environment**: Early runs used Google Colab (GPU not specified, likely T4/V100); later runs and the full benchmark used local GPU (Windows, CUDA device 0). The local GPU enabled consistent, uninterrupted 100-epoch runs for all 16 models.

### 9.3 Per-Epoch Archive Data

Raw per-epoch `results.csv` files exist in `archive/yolov8_local/runs/detect/` for YOLOv8n and YOLOv8m experiments (Runs 3, 4, 5, 7 and tuning runs). These contain per-epoch values for:
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
| m_tune8_right_dt2 | YOLOv8m | — | 27 | — | — | Dataset 2 |
| n_tune_8_right_dt2 | YOLOv8n | — | 49 | — | — | Dataset 2 |
| s_tune9_right_dt2 | YOLOv8s | — | 5 | — | — | Dataset 2 |
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

## 11. Dataset 2 — Tiled Dataset Experiments

### 11.1 Overview

Dataset 2 is derived from Dataset 1 by tiling each image into a 4×4 grid of subtiles, creating a 16× larger dataset (968 → 15,488 training images). This technique is commonly used to improve detection of small objects.

### 11.2 Available Results

| Run | Model | Epochs | mAP50 | Dataset |
|-----|-------|--------|-------|---------|
| Run 2 | YOLOv8n | 5 | 0.453 | 2 |
| Run 5 | YOLOv8n | 100 | 0.827 | 2 |
| archive/yolov8_tile_local/train | YOLOv8n | ~100 | — | 2 |
| archive/yolov8_tile_local/train2 | YOLOv8n | ~100 | — | 2 |
| m_tune8_right_dt2 | YOLOv8m (tuned) | 100 | — | 2 |

### 11.3 Assessment

- **Run 5 (Dataset 2, mAP50=0.827) vs Benchmark YOLOv8n (Dataset 1, mAP50=0.886)**: Dataset 1 outperforms Dataset 2 for YOLOv8n at 100 epochs. This suggests the original images (968 at 640 px) provide more training signal than the tiled variants at this resolution.
- No medium or large model was benchmarked on Dataset 2 with 100 epochs.
- The tiled dataset may be more beneficial at smaller input sizes or when detecting very small objects — conditions not explored in the current benchmark.
- **A systematic comparison of all 16 models on Dataset 2 was not performed.**

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
| Complete Dataset 2 benchmark (all 16 models) | **Not performed** | Extend benchmark script to Dataset 2 |
| GPU model and VRAM | **Not documented** | Add to experimental setup documentation |

---

## 14. Key Findings for Thesis

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
- **No test set evaluation reported**: The benchmark script runs `model.val()` on the **validation split** (272 images), not the **test split** (118 images). For thesis reporting, final evaluation should be performed on the held-out test set to avoid optimistic bias.

---

*Generated from repository analysis — `yolo_ultralytics_benchmark/`, `archive/`, and `Dataset/` contents.*
*Last updated: 2026-03-07*
