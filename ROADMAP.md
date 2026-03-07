# Project Roadmap — Remaining Technical Work

> Current state: 16 models trained for 100 epochs on Dataset 1 (val split metrics only).
> Scope: `yolo_ultralytics_benchmark/` only.
> Goal: Complete the experimental pipeline for thesis submission.

---

## Phase 1 — Test Set Evaluation — DONE ✓

> Completed 2026-03-07. Results: `yolo_ultralytics_benchmark/results/test_results.csv`
> Script: `yolo_ultralytics_benchmark/scripts/test_evaluation.py`

---

## ~~Phase 1 — Test Set Evaluation (Priority: CRITICAL) `~1 hour`~~

**Why**: All current metrics come from the **validation split** (272 images). The **test split** (118 images) is held-out data that the models have never seen. A thesis must report results on the test set — val-set numbers are considered optimistic.

**What to do**: Run `model.val(split='test')` for all 16 `best.pt` checkpoints.

**Output**: mAP50, mAP50-95, Precision, Recall, Speed (ms/image) on the test set — a direct counterpart to the existing benchmark table.

> Speed (ms/image) comes for free from `model.val()` output — covers Phase 2 simultaneously.

---

## Phase 2 — Inference Speed (included in Phase 1) `~0 extra hours`

Ultralytics prints `Speed: X ms preprocess / Y ms inference / Z ms postprocess` at the end of every `val` call. Record the **inference ms/image** for each of the 16 models from the Phase 1 runs.

This enables an **accuracy vs latency** plot for the thesis.

---

## Phase 3 — Choose the Best Model `~0 hours`

After Phase 1, pick the final model using this decision tree:

```text
Is recall the priority?  (conservation → every missed turtle matters)
    YES → YOLO11m  (expected highest recall)

Is a single strongest model needed regardless of speed?
    YES → YOLOv9c  (expected highest mAP50)

Is real-time speed a hard constraint (≥ 25 FPS)?
    YES → compare Phase 2 results; if YOLOv9c is too slow, use YOLOv8m or YOLO11m
```

---

## Phase 4 — ByteTrack Integration — DONE ✓

> Completed 2026-03-07. Scripts: `yolo_ultralytics_benchmark/scripts/tracking.py` and `bytetrack_turtles.yaml`
> Model selected: YOLOv9m (best test mAP50/mAP50-95 balance, 11.9 ms/img, ~84 FPS end-to-end headroom)

---

## ~~Phase 4 — ByteTrack Integration (Priority: HIGH) `~4–6 hours`~~

**Why**: Detection alone identifies turtles frame by frame. Tracking assigns a **persistent ID** to each individual across frames — essential for population counting and re-identification in video surveys.

### 4.1 — Do we need video annotations?

**Short answer: No, not for a first working implementation.**

Here is the distinction:

| Goal | Needs video GT annotations? | What the existing annotations give |
| --- | --- | --- |
| Run ByteTrack and visualise results | **No** | Enough — use images as a sequence or run on raw video |
| Measure FPS, ID-switch count (visual) | **No** | Count switches manually by inspecting output video |
| Compute MOTA / HOTA formally | **Yes** | Existing per-image bbox annotations have **no track IDs**, so MOTA/HOTA cannot be computed from them |

**Why can't the existing image annotations be used for MOTA/HOTA?**
The dataset has one `.txt` label per image with bounding boxes, but there is no link between "box #2 in frame 10" and "box #1 in frame 11" — i.e., no track IDs across frames. MOTA and HOTA require ground-truth track IDs to count ID switches correctly.

**Practical approach for thesis (no video GT needed):**

1. Run ByteTrack on a drone video or image sequence from the dataset.
2. Record output video with overlaid track IDs.
3. Visually count ID switches and track losses.
4. Measure end-to-end FPS.
5. Report qualitative results + FPS as the tracking evaluation.

If formal MOTA/HOTA is required, video sequences would need to be annotated with consistent track IDs across frames — that is a separate annotation effort.

### 4.2 — Step-by-step

#### Step 1: Run tracking on a video or image folder

```bash
yolo track \
  model=yolo_ultralytics_benchmark/models/yolov9c.pt \
  source=<path_to_video_or_folder_of_frames> \
  tracker=bytetrack.yaml \
  conf=0.3 \
  iou=0.5 \
  save=True \
  project=runs/track \
  name=yolov9c_bytetrack
```

#### Step 2: Tune ByteTrack parameters

Copy the default tracker config and adjust for the sea turtle use case:

```bash
cp .venv/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml bytetrack_turtles.yaml
```

Key parameters:

| Parameter | Default | Recommended | Reason |
|---|---|---|---|
| `track_high_thresh` | 0.5 | 0.35 | Accept slightly lower-confidence detections into tracks |
| `track_low_thresh` | 0.1 | 0.1 | Keep default |
| `new_track_thresh` | 0.6 | 0.5 | Lower threshold to start new track faster |
| `track_buffer` | 30 | 60–90 | Turtles can submerge briefly; keep track alive for 2–3 s at 30 FPS |
| `match_thresh` | 0.8 | 0.8 | Keep default; aerial view = consistent appearance |
| `frame_rate` | 30 | match your video FPS | Affects Kalman filter timing |

#### Step 3: Measure FPS

```python
from ultralytics import YOLO
import time

model = YOLO("yolo_ultralytics_benchmark/models/yolov9c.pt")
results = model.track(source="<video>", tracker="bytetrack_turtles.yaml", stream=True)

t0 = time.time()
for i, r in enumerate(results):
    pass
fps = i / (time.time() - t0)
print(f"End-to-end FPS: {fps:.1f}")
```

#### Step 4: Record what to report in thesis

| Metric | How to measure |
| --- | --- |
| End-to-end FPS | Step 3 above |
| Number of unique track IDs | `len(set of IDs seen in output)` |
| Visual ID switches | Inspect output video; count moments when a turtle's ID changes |
| Track loss on submersion | Note sequences where a turtle disappears and check if same ID is recovered |
| Confidence distribution of tracked detections | Check output `.txt` files |

---

## Phase 5 — Hyperparameter Tuning (Priority: LOW / Optional) `~20–75 hours`

**Context**: Previous archive tuning experiments did not improve over Ultralytics defaults. Given that the benchmark results (mAP50 > 0.955 for YOLOv9c) are already strong, tuning is unlikely to yield meaningful gains for this dataset.

Only recommended if Phase 1 reveals a **gap > 0.010 mAP50** between val and test results (sign of overfitting to val), which would justify retraining with adjusted augmentation.

**Candidates if pursued**: YOLOv9c, YOLO11m.

**Effort**: ~20 iterations × ~90 min = ~30 hours per model. High cost, low expected return.

---

## Execution Order

```
Phase 1+2  →  Phase 3  →  Phase 4  →  Phase 5 (only if needed)
 ~1 hour      instant      ~4–6 h       optional
```

## Milestone Checklist

- [x] Phase 1: Run `model.val(split='test')` for all 16 models
- [x] Phase 1: Save test-set metrics to `yolo_ultralytics_benchmark/results/test_results.csv`
- [x] Phase 2: Record inference ms/image per model (from Phase 1 output)
- [x] Phase 3: Select final model for ByteTrack (YOLOv9m)
- [x] Phase 4a: Create `bytetrack_turtles.yaml` with tuned parameters
- [x] Phase 4b: Create `tracking.py` — runs ByteTrack on video or image folder, saves annotated output and `tracking_stats.txt`
- [ ] Phase 4c: Run tracker on actual video/sequence and record FPS, qualitative ID-switch analysis
- [ ] Phase 5: (optional) Tune top-1 or top-2 models if gap found

---

Roadmap updated: 2026-03-07 (Phase 4 infra complete)
