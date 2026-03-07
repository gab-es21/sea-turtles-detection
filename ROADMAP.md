# Project Roadmap — Remaining Technical Work

> Current state: 16 models trained for 100 epochs on Dataset 1 (val split metrics only).
> Goal: Complete the experimental pipeline for thesis submission.

---

## Phase 1 — Proper Test Set Evaluation (Priority: CRITICAL)

**Why**: All current metrics come from the **validation split** (272 images). The **test split** (118 images) has never been evaluated. A thesis must report results on held-out test data — otherwise the reported numbers are optimistic.

**What to do**: Run `model.val(split='test')` for all 16 `best.pt` checkpoints.

**Expected output**: mAP50, mAP50-95, Precision, Recall on test set — comparable table to the current one.

**Effort**: Low — the trained weights already exist; this is purely an evaluation pass.

---

## Phase 2 — Inference Speed Measurement (Priority: HIGH)

**Why**: For a real deployment (drone-side processing or field laptop), knowing the latency per frame is essential. Currently only training time is reported.

**What to do**: For each of the 16 models, measure average inference time on the test set.

```bash
yolo val model=<model>/best.pt data=Dataset/sea-turtles-1/data.yaml split=test
# or
python -c "from ultralytics import YOLO; m=YOLO('best.pt'); m.val(data='...', split='test')"
```

Ultralytics reports `Speed: preprocess / inference / postprocess ms/image` in the val output.

**Expected output**: Inference time (ms/image) per model — enables a proper accuracy vs latency trade-off plot.

**Effort**: Low — same as Phase 1, just read the speed line from output.

---

## Phase 3 — Hyperparameter Tuning on Top-3 Models (Priority: MEDIUM)

**Why**: Previous tuning (archive) used short trials (~25–50 epochs per iteration). A properly configured tuning run at 100 epochs/trial may still improve results marginally. Given the near-saturation of mAP50, the main benefit would be for **recall** on the test set.

**Recommended candidates**: YOLOv9c, YOLO11m (highest recall), YOLO26m (best localisation).

**What to do**:
```python
model = YOLO("yolov9c.pt")
model.tune(
    data="Dataset/sea-turtles-1/data.yaml",
    epochs=100,
    iterations=50,
    imgsz=640,
    batch=8,
    seed=42,
    device=0,
)
```

**Key lesson from archive**: Previous tuning did not beat defaults. Only worth doing if there is time — do not block Phase 4 on this.

**Expected output**: Possibly +0.005–0.010 mAP50 gain; more likely a confirmation that defaults are near-optimal.

**Effort**: High (50 iterations × ~90 min each = ~75 hours per model). Reduce to 20 iterations if time-constrained.

---

## Phase 4 — ByteTrack Integration (Priority: HIGH)

**Why**: The end application is tracking individual turtles across video frames (population monitoring). Detection alone is not sufficient — we need persistent IDs per individual.

**Model to use**: Best model from Phase 1 test evaluation. Expected to be **YOLOv9c** (highest mAP50) or **YOLO11m** (highest recall — fewer missed = fewer ID losses).

### Step 4a — Basic tracking test

```bash
yolo track \
  model=yolo_ultralytics_benchmark/models/yolov9c.pt \
  source=<path_to_video_or_image_sequence> \
  tracker=bytetrack.yaml \
  conf=0.3 \
  iou=0.5 \
  save=True
```

### Step 4b — ByteTrack parameter tuning

Key parameters to sweep (edit `.venv/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml`):

| Parameter | Default | Suggested range | Effect |
|---|---|---|---|
| `track_high_thresh` | 0.5 | 0.3 – 0.6 | Min confidence for primary association |
| `track_low_thresh` | 0.1 | 0.05 – 0.2 | Min confidence for secondary association |
| `new_track_thresh` | 0.6 | 0.4 – 0.7 | Min confidence to initialise new track |
| `track_buffer` | 30 | 30 – 90 | Frames to keep lost track alive (for submersion) |
| `match_thresh` | 0.8 | 0.7 – 0.9 | IoU threshold for track-detection matching |

For sea turtles that can submerge and reappear, **increase `track_buffer`** (e.g., 60–90 frames at 25 FPS = 2.4–3.6 s of memory).

### Step 4c — Metrics to measure

| Metric | Description |
|---|---|
| MOTA | Multi-Object Tracking Accuracy |
| HOTA | Higher Order Tracking Accuracy |
| ID switches (IDSW) | Times a turtle's ID changes |
| Track fragmentation | How often a track is interrupted |
| End-to-end FPS | Detection + tracking throughput |

**Effort**: Medium — requires annotated video sequences (ground-truth tracks) for MOTA/HOTA. If no GT video annotations exist, qualitative visual inspection of tracked output is the fallback.

---

## Summary Roadmap

```
Phase 1 — Test set evaluation       [~1 hour]    CRITICAL
Phase 2 — Inference speed           [~1 hour]    HIGH
Phase 3 — Hyperparameter tuning     [~75 hours]  MEDIUM (optional)
Phase 4a — ByteTrack basic test     [~2 hours]   HIGH
Phase 4b — ByteTrack param tuning   [~4 hours]   MEDIUM
Phase 4c — Tracking metrics         [depends on GT annotations]
```

### Recommended execution order

```
Phase 1 → Phase 2 → Phase 4a → Phase 4b → Phase 3 (if time allows)
```

Run Phases 1 and 2 first because they are fast and complete the evaluation of existing work.
Start Phase 4 (ByteTrack) as soon as the best model is confirmed from Phase 1.
Only invest in Phase 3 (tuning) if the Phase 1 results reveal a clear gap to close.

---

## Decision Tree — Choosing the Final Model for ByteTrack

```
Phase 1 test results available?
    |
    ├─ YOLOv9c still best mAP50 on test set?
    │       └─ YES → use YOLOv9c for tracking
    │
    ├─ Is recall the priority? (conservation = minimise missed turtles)
    │       └─ YES → use YOLO11m (highest recall)
    │
    └─ Is real-time FPS a hard constraint?
            └─ YES → check Phase 2 results; if YOLOv9c < 25 FPS, fall back to YOLOv8m or YOLO11m
```

---

*Roadmap created: 2026-03-07*
