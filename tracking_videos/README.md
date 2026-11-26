# Tracking videos

Side-by-side comparison of the four detector + tracker configurations on the
Dataset 3 drone videos (Noguchi et al., 2025). Each clip is a 2×2 grid showing
the same seconds of the same video, in sync:

| Panel | Configuration |
| --- | --- |
| (a) top left | Exp. 3, Model A (YOLO26s pre-trained) + BoT-SORT |
| (b) top right | `turtle.pt` (authors' YOLOv7) + BoT-SORT |
| (c) bottom left | Exp. 1, YOLOv9m + ByteTrack |
| (d) bottom right | Exp. 4 (YOLO26s, full Dataset 3) + BoT-SORT |

| File | Video | Seconds | Scene |
| --- | --- | --- | --- |
| `grid_No5_t42-53.mp4` | No5 | 42–53 | simple |
| `grid_No5_t150-170.mp4` | No5 | 150–170 | simple, all models follow the same turtle |
| `grid_No6_t125-154.mp4` | No6 | 125–154 | simple |
| `grid_No7_t100-110.mp4` | No7 | 100–110 | hard, many false tracks |
| `grid_No5_No7_simple_vs_hard.mp4` | No5 + No7 | 8 s + 10 s | simple scene followed by hard scene |

Built from the clips in `videos_to_share/` with ffmpeg (`xstack`, 960×540 per panel).
