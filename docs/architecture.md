# SwingScan Architecture

This is the V1 architecture reference. For the stage-by-stage build
history, see `docs/plans/stageN-status.md`; for operating principles,
see `CLAUDE.md`.

## High-level data flow

```
                              ┌────────────────────────┐
   swing video (mp4)  ───►   │  io/video.VideoReader   │
                              └──────────┬─────────────┘
                                         │  BGR uint8 frames
                                         ▼
                              ┌────────────────────────┐
                              │ pose/mediapipe_backend │
                              │ MediaPipePoseEstimator │
                              └──────────┬─────────────┘
                                         │  PoseSequence
          ┌──────────────────────────────┼──────────────────────────────┐
          │                              │                              │
          ▼                              ▼                              ▼
┌──────────────────┐           ┌────────────────────┐          ┌──────────────────┐
│ club/detector    │           │ phases/segmenter   │          │ io/serialize     │
│ HeuristicClub    │           │ HeuristicSegmenter │          │ save/load parquet │
│ (fallback) or    │           │ (fallback) or      │          └──────────────────┘
│ YoloClubDetector │           │ SwingNetSegmenter  │
└────────┬─────────┘           └─────────┬──────────┘
         │  ClubDetection[]              │  PhaseMap
         ▼                               │
┌──────────────────┐                     │
│ club/tracker     │                     │
│ ClubTracker      │                     │
│ smooth + gapfill │                     │
└────────┬─────────┘                     │
         │  ClubTrack                    │
         └───────────────┬───────────────┘
                         │
                         ▼
                 ┌─────────────────────┐
                 │ metrics/biomech     │
                 │ compute_swing_...   │
                 └──────────┬──────────┘
                            │  SwingMetrics
                            ▼
                 ┌─────────────────────┐     ┌───────────────────┐
                 │ compare/diff        │◄────│ compare/pro_bank  │
                 │ SwingDiff           │     │ ProBank.load      │
                 └──────────┬──────────┘     └───────────────────┘
                            │  SwingDiff
                            ▼
                 ┌─────────────────────┐     ┌───────────────────┐
                 │ feedback/rules      │◄────│ configs/          │
                 │ RuleEngine.evaluate │     │ feedback_rules.yaml│
                 └──────────┬──────────┘     └───────────────────┘
                            │  FeedbackItem[]
                            ▼
                 ┌─────────────────────┐
                 │ feedback/render     │ → text / JSON payload
                 │ viz/overlay         │ → annotated mp4
                 └─────────────────────┘
```

Every box is a concrete Python module. Every edge is a typed
dataclass or protocol — nothing flows through `dict[str, Any]` by
convention.

## Module responsibilities

| Module | Role | Stage |
|---|---|---|
| `io/video.py` | Frame iterator with rotation + ffmpeg checks. | 1 |
| `io/serialize.py` | `PoseSequence` ↔ parquet / JSON. | 1 |
| `pose/keypoints.py` | `Joint` enum + semantic groupings. | 1 |
| `pose/base.py` | `PoseFrame`, `PoseSequence`, `PoseEstimator` protocol. | 1 |
| `pose/mediapipe_backend.py` | BlazePose wrapper. | 1 |
| `club/detector.py` | `ClubDetection`, `HeuristicClubDetector`, `YoloClubDetector`. | 2 |
| `club/tracker.py` | `ClubTrack` with gap-fill + EMA smoothing. | 2 |
| `phases/events.py` | `SwingEvent` enum. | 3 |
| `compare/pro_bank.py` | `ProBank` store, `SwingLabel`, `build_pro_bank`. | 3 |
| `phases/segmenter.py` | `HeuristicSegmenter` + protocol. | 4 |
| `phases/swingnet.py` | SwingNet weight-gated stub. | 4 |
| `metrics/angles.py` | Per-frame joint-angle primitives. | 5 |
| `metrics/normalize.py` | Scale + recenter + handedness flip. | 5 |
| `metrics/biomech.py` | `PhaseMetrics`, `SwingMetrics`. | 5 |
| `compare/diff.py` | `SwingDiff` against a cohort with z-scores. | 5 |
| `feedback/rules.py` | YAML rule engine, `FeedbackItem`. | 6 |
| `feedback/render.py` | Text + JSON renderers with cap. | 6 |
| `viz/overlay.py` | Skeleton / club trail / phase banner → mp4. | 7 |
| `pipeline.py` | Top-level orchestration used by CLI + demo. | 2/6/7 |
| `cli.py` | `version` / `pose` / `phases` / `run` subcommands. | 0+ |

## Backend fallbacks

Three subsystems have a primary ML backend and a no-weights fallback:

* **Club detection:** YOLOv8 → pose-derived geometric proxy.
* **Phase segmentation:** SwingNet → wrist-velocity heuristic.
* **Pro bank:** real GolfDB-sourced bank → empty bank (degrades to
  metrics-without-cohort-deltas).

The fallbacks exist so the V1 pipeline has zero hard external
downloads on first run. Each is tagged with an ADR when the fallback
has non-obvious tradeoffs (see `docs/decisions/`).

## Performance budget (V1 targets)

Measured informally on the synthetic Stage 1 fixture (160×160, 60
frames) on CPU:

- Full `swingscan run` with heuristic club + no pro bank: **~2 s** for
  a 2-second clip (dominated by MediaPipe warmup + a second
  `VideoReader` pass for the club loop).
- Annotated video render (`--output-video`): **<1 s** additional cost.
- Unit + integration suite: **~13 s** for 104 tests.

The CLAUDE.md §2 target ("under 60 seconds on a single consumer GPU
for a clean face-on driver clip") is comfortably inside this budget
on CPU for a 5-second clip at 1080p once MediaPipe's per-frame cost
dominates.

## Extension points for V2

See [`ROADMAP.md`](../ROADMAP.md).
