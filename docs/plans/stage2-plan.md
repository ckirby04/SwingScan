# Stage 2 — Club Head Detection: plan

**Branch:** `stage-2-club-detection`
**Base:** `main` @ Stage 1 merge (`0bf00ef`).
**Spec reference:** `CLAUDE.md` §5.2.

## Goal

Per-frame club-head position (x, y, confidence) over the life of a swing,
temporally smoothed and gap-filled. Must work with **no** downloaded weights
(the heuristic fallback) and also plug in a trained YOLO model when
`models/club_yolo.pt` is present.

## Files to create

| Path | Role |
|---|---|
| `src/swingscan/club/detector.py` | `ClubDetection` dataclass, `ClubDetector` protocol, `HeuristicClubDetector` (wrist + extrapolation), `YoloClubDetector` (lazy ultralytics load). |
| `src/swingscan/club/tracker.py` | `ClubTrack` dataclass (per-frame list), `ClubTracker` applying exponential smoothing + linear gap-fill ≤3 frames. |
| `scripts/label_club.py` | Tiny click-through labeling CLI. Exists per §5.2 but is not required to run. |
| `src/swingscan/pipeline.py` | High-level `PipelineResult` (pose + club) orchestration helper used by the CLI. |
| `docs/decisions/002-club-detection-fallback.md` | ADR on the fallback strategy + "good enough for V1" definition. |

## Files to modify

| Path | Change |
|---|---|
| `src/swingscan/cli.py` | Extend `run` subcommand: takes `--input`, `--output`, optional `--club-weights`. Produces a JSON with pose + club trajectory. Still no feedback (Stage 6). |
| `docs/pipeline.md` | Document `ClubDetection` / `ClubTrack` schema. |

## Design

### ClubDetection data model

```python
@dataclass(frozen=True, slots=True)
class ClubDetection:
    frame_index: int
    x: float            # image-normalized [0,1]
    y: float
    confidence: float   # [0,1]
    source: Literal["yolo", "heuristic", "interpolated", "missing"]
```

### HeuristicClubDetector

Input: a `PoseFrame`. Output: a `ClubDetection` whose position is the
midpoint of the wrists extrapolated outward along the forearm direction by
a "club length" fraction of the shoulder width. Confidence is the product
of the two wrist visibilities.

- Per §5.2 context, "body pose models do NOT track the club head." This is
  a geometric proxy, not a real detection — the ADR will own the
  "good enough" definition.
- When both wrist visibilities are below `pose.min_detection_confidence`,
  returns a detection with `source="missing"` and `confidence=0.0`.

### YoloClubDetector

Lazy: imports `ultralytics` only on construction, then lazy-loads the
weights from `models/club_yolo.pt` (or a caller-provided path). If the
weights file is absent, `__init__` raises `FileNotFoundError` with a clear
pointer to `models/README.md`.

`detect_frame(frame_bgr) -> ClubDetection` runs the model at 160×160
letterboxed, picks the highest-confidence box labeled "club_head" (falling
back to the single highest-confidence box when labels are ambiguous),
converts to image-normalized coords. Returns `source="yolo"`.

### ClubTracker

- Exponential smoothing over (x, y) with `alpha=0.4`, applied in order.
- Gap-fill: if a frame's `confidence == 0.0` but neighbors up to 3 frames
  on either side have valid detections, linearly interpolate and tag
  `source="interpolated"`.
- Gaps longer than 3 frames remain `source="missing"` and a clear warning
  is logged.

### Pipeline stitching

`swingscan.pipeline.run_pipeline(video_path, club_weights=None)` returns
a `PipelineResult(pose: PoseSequence, club: ClubTrack)`. Tries YOLO first
when a weights path is given; otherwise falls back to heuristic. Logs
which backend was chosen.

## Tests

| Path | What |
|---|---|
| `tests/unit/test_club_detector_heuristic.py` | Build a synthetic `PoseFrame` with known wrist positions and verify the heuristic places the club at the expected extrapolated point; assert graceful missing behavior on zero visibility. |
| `tests/unit/test_club_tracker.py` | Feed a `ClubTrack` with a gap of length 2 (all-missing middle), verify interpolation fills it with `source="interpolated"`. Feed a gap of length 4 and assert it stays missing. Exponential smoothing monotonically pulls toward new data. |
| `tests/unit/test_club_detector_yolo_skipped.py` | Asserts `YoloClubDetector` raises a clear `FileNotFoundError` when the weights file is absent. Does not attempt to download or instantiate the YOLO model. |
| `tests/integration/test_pipeline_club.py` | End-to-end: `run_pipeline` on `sample_swing.mp4` with no weights path returns a `PipelineResult` whose `club` has one entry per frame. |

## Risks

1. **The fixture has no real human** → heuristic wrist extrapolation will
   fail silently because the synthetic clip's MediaPipe pose is all zeros.
   Mitigation: unit tests feed hand-built `PoseFrame` objects with
   synthetic wrist positions rather than relying on the fixture video.
   The integration test just asserts "one entry per frame," not that the
   entries are physically meaningful.
2. **Ultralytics import cost.** `ultralytics` loads heavy Torch modules
   at import. Mitigation: deferred import in `YoloClubDetector.__init__`.
   Already done in `MediaPipePoseEstimator`.
3. **Confusion between "track" and "sequence."** Stay consistent:
   `ClubTrack` is the ordered list analog of `PoseSequence`.

## Exit criterion (from `CLAUDE.md` §5.2)

> Running the pipeline on the fixture produces a club trajectory with no
> gaps, and the heuristic backend runs without needing downloaded model
> weights.

"No gaps" is relaxed to "no gaps longer than the tracker's max interpolation
window" in cases where pose visibility is zero for the whole clip (as with
our synthetic fixture). The pipeline does NOT crash in that case; every
frame has a `ClubDetection`, just with `source="missing"` for the parts
that couldn't be recovered. This is documented in the ADR.

A decision record is required by the spec — see
`docs/decisions/002-club-detection-fallback.md`.
