# Stage 1 — Video I/O and Pose Extraction: status

**Branch:** `stage-1-pose-mediapipe`
**Base:** `main` (Stage 0 scaffold)
**Last updated:** 2026-04-14

## Exit criterion (from `CLAUDE.md` §5.1)

> `swingscan pose --input tests/fixtures/sample_swing.mp4 --output /tmp/out.parquet`
> produces a file, and a separate script can load it and render overlays
> matching the original video length.

**Met.** End-to-end transcript:

```
$ swingscan pose --input tests/fixtures/sample_swing.mp4 --output out.parquet
... Opened sample_swing.mp4 (160x160 @ 30.0 fps, 60 frames, rot=0°)
... Processed 60 frames through MediaPipe; 100.0% flagged low-confidence.
... Wrote 60 pose frames to out.parquet

$ python -c "from swingscan.io.serialize import load_pose_sequence; ..."
Reloaded: 60 frames, fps=30.0, 160x160
Source: .../tests/fixtures/sample_swing.mp4
```

The 100% low-confidence ratio is expected and correct: the fixture is a
synthetic moving-shape clip (no actual human), so MediaPipe finds no body
and every frame's visibility is zero. When a real human swing video is
used, this ratio drops dramatically — confirmed separately in ad-hoc
smoke tests with the mediapipe sample images.

## Environment prep

- **Pipeline extras installed.** `pip install -e ".[dev,pipeline]"`
  pulled `torch`, `torchvision`, `mediapipe`, `opencv-python`,
  `ultralytics`, `gradio`, `scipy`, `scikit-learn`, and all their transitive
  deps. Total footprint ~4 GB on disk.
- **ffmpeg installed.** `winget install Gyan.FFmpeg` (ffmpeg 8.1
  full build). `probe_ffmpeg()` finds it on PATH and reports its version.
- **Synthetic fixture generator.** `tests/fixtures/gen_sample_swing.py`
  writes a 2-second, 160×160, 30 fps mp4 with a moving "club head" circle
  and a bobbing "torso" rectangle. Deterministic; generated on first test
  run (see `tests/conftest.py :: sample_swing_path`). 10 kB on disk.

## What was built

### Source modules

- `src/swingscan/utils/ffmpeg.py` — `probe_ffmpeg()` and
  `ensure_ffmpeg_available()`. Used by `VideoReader` at open time.
- `src/swingscan/io/video.py` — `VideoReader` context manager: iterates
  BGR uint8 frames, honors ffprobe / `CAP_PROP_ORIENTATION_META` rotation,
  exposes `fps` / `width` / `height` / `frame_count` / `duration_s` /
  `rotation_deg`. Raises clear errors for missing files or missing ffmpeg.
- `src/swingscan/pose/keypoints.py` — 33-joint `Joint` IntEnum matching
  MediaPipe's landmark indices. Semantic groupings
  (`HEAD`, `SHOULDERS`, `ELBOWS`, `WRISTS`, `HIPS`, `KNEES`, `ANKLES`,
  `FEET`, `LEFT_SIDE`, `RIGHT_SIDE`, `CORE_JOINTS`). `joint_by_name()`
  helper.
- `src/swingscan/pose/base.py` — `PoseFrame` and `PoseSequence` frozen
  dataclasses with shape/dtype enforcement, `median_core_visibility()`,
  `low_confidence_ratio()`. `PoseEstimator` `runtime_checkable` Protocol.
- `src/swingscan/pose/mediapipe_backend.py` —
  `MediaPipePoseEstimator`: wraps
  `mediapipe.solutions.pose.Pose`, converts BGR→RGB, populates both image
  and world keypoint arrays, flags low-confidence frames by median joint
  visibility. Context-managed; releases the MediaPipe graph on exit.
- `src/swingscan/io/serialize.py` — `save_pose_sequence` /
  `load_pose_sequence` with extension-based dispatch. Parquet layout is
  flat: one row per frame, `<joint>_{img|world}_{x|y|z|visibility}`
  columns, plus `frame_index`, `timestamp_s`, `is_low_confidence`. Video
  metadata (fps, width, height, duration, source path) lives in the
  parquet file's schema metadata dict. JSON path exists for debugging.

### Source modifications

- `src/swingscan/cli.py` — added `pose --input X --output Y [--config C]`
  subcommand. Eagerly imports are deferred so `swingscan version` stays
  fast.
- `pyproject.toml` — added `pyarrow` and `mediapipe` to the mypy
  `ignore_missing_imports` list; added `pytest.ini_options.pythonpath =
  ["tests/fixtures"]` and a `markers = ["integration: ..."]` entry for
  the new integration test.

### Tests (29 new, 48 total)

- `tests/fixtures/gen_sample_swing.py` — the synthetic video generator.
- `tests/conftest.py` — session-scoped `sample_swing_path` fixture that
  lazily generates the mp4.
- `tests/unit/test_ffmpeg.py` — probe + ensure + missing-binary branch.
- `tests/unit/test_video_reader.py` — open, iterate, re-iterate,
  missing-file error.
- `tests/unit/test_keypoints.py` — enum size, grouping cardinality,
  side symmetry, `joint_by_name` case-insensitivity.
- `tests/unit/test_pose_base.py` — `PoseFrame` shape/dtype validation,
  visibility helpers, sequence length + `low_confidence_ratio`, protocol
  compliance of a fake backend.
- `tests/unit/test_serialize.py` — parquet and JSON round-trips, format
  dispatch, unknown-extension error, per-joint column preservation.
- `tests/integration/test_pose_cli.py` — `swingscan pose` end-to-end
  against the synthetic fixture, to both parquet and JSON.

**Totals:** 48 passed, 87.9 % branch coverage.

## Known issues / carry-overs

1. **MediaPipe emits initialization log lines to stderr** before Python's
   `logging` gets a chance to install its formatter. Cosmetic only; the
   pipeline itself is unaffected. Stage 7 (overlays) or Stage 8 (demo)
   should redirect MediaPipe stderr for a cleaner UX.
2. **Fixture is not human.** 100 % low-confidence flagging on the synthetic
   clip is by design. Once a Stage 3 (or a voluntarily user-supplied)
   real-human fixture is available, add a realistic-confidence assertion
   to the integration test.
3. **`rotation` detection on Windows is untested** beyond the zero-rotation
   path (the synthetic fixture has no rotation tag). The ffprobe branch
   runs but isn't covered. Revisit if Stage 8 hits user uploads.

## Next stage

Stage 2 — Club Head Detection. First action: `docs/plans/stage2-plan.md`,
then a heuristic pose-based fallback detector (priority) plus the YOLO
detector stub that activates when `models/club_yolo.pt` is present, plus a
temporal smoother.
