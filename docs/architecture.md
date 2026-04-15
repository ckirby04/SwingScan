# SwingScan Architecture

For stage-by-stage build history see `docs/plans/stageN-status.md`;
for operating principles see `CLAUDE.md`. This file is the current
architecture reference.

## High-level data flow

```
   swing video (mp4)
         │
         ▼
 ┌────────────────────────┐
 │  io/video              │
 │  VideoReader           │   BGR frames (uint8, HxWx3)
 │  ffmpeg rotation probe │
 └──────────┬─────────────┘
            │
            ▼
 ┌────────────────────────┐
 │  pose/mediapipe_backend│
 │  MediaPipePoseEstimator│   PoseSequence (per-frame 33-joint
 └──────────┬─────────────┘    image + world keypoints)
            │
            ├─────────────────────┐──────────────────────────┐
            │                     │                          │
            ▼                     ▼                          ▼
 ┌──────────────────┐   ┌────────────────────┐      ┌──────────────────┐
 │ club/detector    │   │ phases/swingnet    │      │ io/serialize     │
 │ HeuristicClub    │   │ SwingNetSegmenter  │      │ save/load parquet│
 │ (hand landmarks) │   │ (vendored upstream)│      └──────────────────┘
 │  or              │   │  or                │
 │ YoloClubDetector │   │ HeuristicSegmenter │
 └────────┬─────────┘   │ (wrist velocity    │
          │             │  fallback)         │
          ▼             └──────────┬─────────┘
 ┌──────────────────┐              │ PhaseMap
 │ club/tracker     │              │
 │ ClubTracker      │              │
 │ smooth + gap-fill│              │
 └────────┬─────────┘              │
          │  ClubTrack             │
          └──────┬─────────────────┘
                 │
                 ▼
       ┌─────────────────────┐
       │ metrics/biomech     │
       │ compute_swing_...   │   SwingMetrics (per-event
       └──────────┬──────────┘    + composite values)
                  │
                  ▼
       ┌─────────────────────┐     ┌───────────────────┐
       │ compare/diff        │◄────│ compare/pro_bank  │
       │ SwingDiff           │     │ ProBank.load      │
       │ (circular stats)    │     └───────────────────┘
       └──────────┬──────────┘
                  │  SwingDiff
                  ▼
       ┌─────────────────────┐     ┌─────────────────────────┐
       │ feedback/rules      │◄────│ configs/                │
       │ RuleEngine.evaluate │     │ feedback_rules.yaml     │
       │ (severity-sorted)   │     │ (coach-voice, 15 rules, │
       └──────────┬──────────┘     │  calibrated thresholds) │
                  │                └─────────────────────────┘
                  │  FeedbackItem[]
                  ▼
       ┌─────────────────────┐
       │ feedback/render     │ ─► plain text / JSON payload
       │ viz/overlay         │ ─► annotated mp4
       └─────────────────────┘
```

Every box is a concrete Python module. Every edge is a typed
dataclass or protocol — nothing flows through `dict[str, Any]` by
convention.

## Module responsibilities

| Module | Role |
|---|---|
| `io/video.py` | Frame iterator with rotation metadata and ffmpeg probe. |
| `io/serialize.py` | `PoseSequence` ↔ parquet / JSON round-trip. |
| `pose/keypoints.py` | `Joint` IntEnum + semantic groupings. |
| `pose/base.py` | Frozen `PoseFrame`, `PoseSequence`, `PoseEstimator` protocol. |
| `pose/mediapipe_backend.py` | BlazePose wrapper. |
| `club/detector.py` | `ClubDetection`, hand-landmark `HeuristicClubDetector`, weight-gated `YoloClubDetector`. |
| `club/tracker.py` | `ClubTrack` with causal exponential smoothing + linear gap-fill. |
| `phases/events.py` | `SwingEvent` enum (ordered, canonical). |
| `phases/segmenter.py` | `PhaseSegmenter` protocol + `HeuristicSegmenter` (wrist velocity). |
| `phases/swingnet.py` | Real SwingNet inference wrapper (weight-gated). |
| `phases/_swingnet_model.py` | Vendored MobileNetV2 + BiLSTM (CC BY-NC 4.0 attribution). |
| `metrics/angles.py` | Joint-angle primitives (hip / shoulder / x-factor / spine / arm / wrist / knee / head). |
| `metrics/normalize.py` | Scale + recenter + handedness flip. |
| `metrics/biomech.py` | `PhaseMetrics`, `SwingMetrics`, `compute_swing_metrics`. |
| `compare/pro_bank.py` | `ProBank`, `SwingLabel`, `ProSwing`, `build_pro_bank`. |
| `compare/diff.py` | `SwingDiff` with circular-aware mean + z-scores. |
| `feedback/rules.py` | YAML rule engine → `FeedbackItem[]`. |
| `feedback/render.py` | Text + JSON renderers, severity-capped. |
| `viz/overlay.py` | Skeleton / phase banner / opt-in club trail → mp4. |
| `pipeline.py` | `PipelineResult` + `run_pipeline` orchestration. |
| `cli.py` | `version` / `pose` / `phases` / `run` subcommands. |
| `utils/logging.py` | Idempotent stdlib logging config. |
| `utils/seeding.py` | Seed Python / NumPy / torch (when present). |
| `utils/paths.py` | Repo-root-anchored path helpers. |
| `utils/ffmpeg.py` | `probe_ffmpeg` / `ensure_ffmpeg_available`. |

## Backend fallbacks

Three subsystems have a primary ML backend and a pose-only fallback:

| Subsystem | Primary | Fallback |
|---|---|---|
| Phase segmentation | `SwingNetSegmenter` when `models/swingnet_1800.pth.tar` is present | `HeuristicSegmenter` (wrist-velocity zero-crossing) |
| Club-head detection | `YoloClubDetector` when `models/club_yolo.pt` is present | `HeuristicClubDetector` (hand-landmark-derived shaft direction) |
| Cohort comparison | `ProBank` built from real GolfDB swings | No diff → metrics-only report |

The fallbacks exist so the V1 pipeline has zero hard external
downloads on first run. Each non-obvious fallback is documented in
`docs/decisions/` as an ADR.

## Performance budget (observed)

On CPU-only (no GPU), against the synthetic Stage 1 fixture (160×160,
60 frames):

- `swingscan pose`: ~1–2 s (dominated by MediaPipe warmup).
- `swingscan run` with heuristic club + no pro bank: ~2 s.
- `swingscan run` with SwingNet + pro bank: ~5 s (SwingNet adds one
  full video pass at ~40 fps through the model).
- `--output-video`: <1 s additional.
- Full `make test` suite (116 tests): ~15 s.

On a real 1080p 5-second phone clip, expect roughly 15–25 s on CPU;
the MediaPipe per-frame cost dominates. Real-time is a V2 goal.

## Extension points for V2

See [`../ROADMAP.md`](../ROADMAP.md).
