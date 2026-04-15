# SwingScan Pipeline Reference

End-to-end data contracts shipped in V1. Type definitions live in
`src/swingscan/` — this doc is a narrative index, not a second source
of truth.

## On-disk artifacts

### Pose parquet (`swingscan pose --output X.parquet`)

One row per frame. Columns:

```
frame_index          int32
timestamp_s          float64
is_low_confidence    bool
<joint>_img_{x,y,z,visibility}       float32
<joint>_world_{x,y,z,visibility}     float32
```

where `<joint>` is the lowercase `Joint` enum name (33 values:
`nose`, `left_eye_inner`, ..., `right_foot_index`). Video metadata
(`fps`, `width`, `height`, `duration_s`, `source_path`) lives in the
parquet schema's key/value metadata dict under the
``swingscan_format: pose_sequence_v1`` marker.

### Pose JSON (`swingscan pose --output X.json`)

Debug-only. Equivalent to the parquet above but nested:

```json
{
  "format": "pose_sequence_v1",
  "metadata": {"fps": 30.0, "width": 160, "height": 160,
               "duration_s": 2.0, "source_path": "..."},
  "frames": [
    {"frame_index": 0, "timestamp_s": 0.0, "is_low_confidence": true,
     "image_keypoints": [[x, y, z, vis], ...],
     "world_keypoints": [[x, y, z, vis], ...]},
    ...
  ]
}
```

### Phase JSON (`swingscan phases --output X.json`)

```json
{
  "format": "swingscan_phases_v1",
  "source": "swing.mp4",
  "frame_count": 60,
  "events": {
    "ADDRESS": 0, "TOE_UP": 14, "MID_BACKSWING": 21,
    "TOP": 29, "MID_DOWNSWING": 30, "IMPACT": 31,
    "MID_FOLLOW_THROUGH": 44, "FINISH": 59
  }
}
```

Frame indices are always monotonically increasing within a single
map.

### Pipeline report JSON (`swingscan run --output X.json`)

```json
{
  "format": "swingscan_pipeline_v1",
  "pose_summary": {
    "fps": 30.0, "width": 160, "height": 160,
    "duration_s": 2.0, "source_path": "...",
    "frame_count": 60, "low_confidence_ratio": 1.0
  },
  "club_track": [{"frame_index": 0, "x": 0, "y": 0,
                  "confidence": 0, "source": "missing"}, ...],
  "summary": {
    "frame_count": 60,
    "club_missing_ratio": 1.0,
    "club_interpolated_ratio": 0.0
  },
  "phases": {"ADDRESS": 0, ...},
  "feedback": [
    {"rule": "...", "phase": "TOP", "metric": "...",
     "severity": 3, "message": "...", "amateur": 0.0,
     "delta": 0.0, "z_score": 0.0}
  ]
}
```

`phases` is present when phase segmentation ran (always, in V1).
`feedback` is present when `--pro-bank` was supplied and the rule
engine fired at least one rule.

### Pro bank parquet (`scripts/build_pro_bank.py`)

One row per `(swing_id, event)`. Columns:

```
swing_id        string
view            string
handedness      string
club            string
event           string
frame_index     int32
<joint>_img_{x,y,z,visibility}       float32
<joint>_world_{x,y,z,visibility}     float32
```

Plus a manifest JSON next to the parquet:

```json
{
  "source_labels": "...",
  "bank_parquet": "...",
  "requested": 20,
  "succeeded": 18,
  "skipped": 2
}
```

### Evaluation report (`scripts/evaluate_pipeline.py`)

```json
{
  "format": "swingscan_evaluation_v1",
  "labels_path": "...",
  "tolerance": 5,
  "swings_total": 20,
  "swings_scored": 18,
  "pce": 0.62,
  "pose_complete_ratio": 0.94,
  "bank_coverage": 0.88,
  "per_swing": [...]
}
```

When the labels file is missing, the report still writes but with a
`"marker": "no_data"` field and null numeric fields.

## Runtime flow

1. `cli.py` parses arguments and calls into `_cmd_pose`, `_cmd_phases`,
   or `_cmd_run` (the most common entry point).
2. `pipeline.run_pipeline()` opens a `VideoReader`, constructs a
   `MediaPipePoseEstimator`, and runs `estimate_video()` to produce a
   `PoseSequence`.
3. **Club detection.** If `--club-weights` is supplied and the file
   exists, a `YoloClubDetector` is used; otherwise the pipeline falls
   back to `HeuristicClubDetector`, which derives shaft direction from
   MediaPipe's hand finger landmarks and scales club length by forearm
   length. The result is one `ClubDetection` per frame.
4. `ClubTracker.track()` produces a smoothed `ClubTrack` with
   short-gap linear interpolation (≤ 3 frames) and causal exponential
   smoothing.
5. **Phase segmentation.** If SwingNet weights are present at
   `models/swingnet_1800.pth.tar` (or supplied via `--swingnet-weights`),
   `SwingNetSegmenter` runs the vendored upstream model and picks the
   8 event frames from per-frame 9-class softmax. Without weights the
   pipeline falls back to `HeuristicSegmenter` (wrist y minimum for
   top, wrist speed argmax for impact, midpoints for intermediates).
6. **Cohort comparison.** If `--pro-bank` was supplied, `ProBank.load()`
   reads the parquet, `compare.diff.compare_against_bank()` produces
   a `SwingDiff` with circular-aware central tendency and z-scores,
   and `RuleEngine.from_yaml(...).evaluate(diff)` returns a
   severity-ordered list of `FeedbackItem`.
7. `pipeline.save_pipeline_result()` writes the JSON report above,
   `pipeline.render_report()` prints the text summary, and
   `viz.overlay.render_annotated_video()` writes the annotated mp4
   when `--output-video` is set (the club trail is hidden by default;
   pass `--draw-club` to include it).

Every step's failure mode is covered by an error-path test in
`tests/`. The pipeline does not silently fall back — it logs a
WARNING when it chooses a degraded backend and skips downstream
stages that need missing inputs.
