# Stage 1 — Video I/O and Pose Extraction (MediaPipe baseline): plan

**Branch:** `stage-1-pose-mediapipe`
**Base:** `main` at Stage 0 scaffold (`b5648a2`).
**Spec reference:** `CLAUDE.md` §5.1.

## Goal

Given an input video, produce a per-frame pose record (33 BlazePose joints)
and round-trip it to and from disk. Exit criterion: `swingscan pose --input
tests/fixtures/sample_swing.mp4 --output /tmp/pose.parquet` produces a
parquet file whose row count equals the video frame count, and a separate
helper can load the parquet back into a `PoseSequence` matching the
original length.

## Files to create

### Source

| Path | Role |
|---|---|
| `src/swingscan/io/video.py` | `VideoReader` — iterate frames, expose fps / width / height / duration, honor rotation metadata, detect ffmpeg availability. |
| `src/swingscan/pose/keypoints.py` | `Joint` IntEnum mirroring the 33 BlazePose landmarks with semantic groupings (`HEAD`, `SHOULDERS`, `ELBOWS`, `WRISTS`, `HIPS`, `KNEES`, `ANKLES`, `FEET`). |
| `src/swingscan/pose/base.py` | `Keypoint` dataclass (`x`, `y`, `z`, `visibility`, `presence`), `PoseFrame` dataclass (`frame_index`, `timestamp_s`, per-joint keypoints in both image and world coordinates, `is_low_confidence` flag), `PoseSequence` dataclass (ordered list of frames plus video metadata), and a `PoseEstimator` `typing.Protocol`. |
| `src/swingscan/pose/mediapipe_backend.py` | `MediaPipePoseEstimator` wrapping `mediapipe.solutions.pose.Pose`. Returns both image-normalized and world coordinates; flags frames with any joint below `min_detection_confidence`. |
| `src/swingscan/io/serialize.py` | `save_pose_sequence` / `load_pose_sequence` for parquet (primary) and JSON (debugging). Round-trip preserves every field. |
| `src/swingscan/utils/ffmpeg.py` | `ensure_ffmpeg_available()` helper that probes for ffmpeg on PATH and raises a clear error pointing at install instructions. Shared across future video scripts. |

### Source — modifications

| Path | Change |
|---|---|
| `src/swingscan/cli.py` | Add `pose` subcommand. Wires `VideoReader` → `MediaPipePoseEstimator` → `save_pose_sequence`. Leaves the Stage 0 `run` stub in place. |
| `docs/pipeline.md` | Document the parquet schema and the `PoseSequence` shape. |

### Tests

| Path | What it exercises |
|---|---|
| `tests/fixtures/gen_sample_swing.py` | Deterministic script that synthesizes a short moving-shape video to `tests/fixtures/sample_swing.mp4`. Run once manually; the generated mp4 lives in the fixtures dir (exempted in `.gitignore`) but we commit it only if it is tiny (<~50 kB) and clearly synthetic. |
| `tests/unit/test_video_reader.py` | Frame count matches ffprobe / cv2 metadata, duration / fps / dims are plausible, iteration yields the same count as `frame_count`, cleanup closes the handle. |
| `tests/unit/test_keypoints.py` | Enum size = 33, groupings partition the joints without overlap within reason, round-trip by name. |
| `tests/unit/test_pose_base.py` | `PoseFrame` / `PoseSequence` construction and `is_low_confidence` flagging. |
| `tests/unit/test_serialize.py` | Round-trip `PoseSequence` through parquet and JSON without loss. Uses a synthetic sequence, not a real video, so it runs without the MediaPipe dependency. |
| `tests/unit/test_mediapipe_backend.py` | Smoke-level only: construct the backend, run it on a 1-frame all-gray image, and assert no crash plus a valid `PoseFrame`. Marked `@pytest.mark.mediapipe` and skipped automatically if `mediapipe` import fails (defensive, even though `[pipeline]` is now a required install). |
| `tests/integration/test_pose_cli.py` | Invokes `swingscan pose --input <fixture> --output <tmp>.parquet`, asserts the file exists and reload matches frame count. Marked `@pytest.mark.integration`. |

## Design notes

### Rotation metadata

Phones often embed a rotation tag in the container that cv2 does NOT apply
automatically. I'll probe the tag via ffprobe (if ffmpeg is available) and
rotate frames in `VideoReader` accordingly. When ffprobe is absent, read
cv2's `CAP_PROP_ORIENTATION_META` (OpenCV 4.6+) and fall back to zero
rotation with a warning log.

### Low-confidence flagging

MediaPipe returns per-landmark `visibility` and `presence`. A frame is
marked `is_low_confidence` when the median visibility across the body
landmarks is below `pose.min_detection_confidence`. Recorded per-frame so
downstream code (Stage 4 heuristic segmenter, Stage 5 metrics) can choose
whether to interpolate over or drop such frames.

### Parquet schema

One row per frame, one column per `(joint, coordinate_space, component)`:

```
frame_index           int32
timestamp_s           float64
is_low_confidence     bool
<joint>_img_x         float32   # 0..1, image-space
<joint>_img_y         float32
<joint>_img_z         float32
<joint>_img_vis       float32
<joint>_world_x       float32   # metric, world-space
<joint>_world_y       float32
<joint>_world_z       float32
```

Flat schema keeps pyarrow happy and makes downstream pandas work trivial.
Metadata (fps, width, height, duration, source filename) is stored in the
parquet file's key/value metadata dict, not as columns.

### `PoseEstimator` protocol

```python
class PoseEstimator(Protocol):
    def estimate(self, frame: NDArray[np.uint8], frame_index: int, timestamp_s: float) -> PoseFrame: ...
    def estimate_video(self, video: VideoReader) -> PoseSequence: ...
```

The `estimate_video` default can be a non-Protocol helper (`estimate_video_default`) that backends call into, keeping the protocol minimal.

### CLI ergonomics

`swingscan pose --input X --output Y [--format parquet|json]`. Default
format is parquet; `.json` extension auto-selects JSON. Also supports
`--overlay-output Z.mp4` as a no-op stub until Stage 7 lands (optional, may
defer).

## Risks

1. **MediaPipe on synthetic fixtures.** Moving shapes don't trigger pose
   detection at all, so `test_mediapipe_backend` asserting a `PoseFrame` with
   valid keypoints may fail. Mitigation: the test only asserts the call
   succeeds and returns a `PoseFrame`; it tolerates all-zero keypoints
   flagged `is_low_confidence=True`. The real confidence test lands only
   when a real human fixture is available.
2. **Parquet + pyarrow strict-warnings interaction.** pytest `filterwarnings =
   error` may elevate a pyarrow future-warning to a failure. Mitigation:
   add a targeted `ignore::FutureWarning:pyarrow.*` filter if I hit it.
3. **cv2 rotation on Windows.** OpenCV's Windows wheel sometimes builds
   without ffmpeg back-end features. Mitigation: the `ensure_ffmpeg_available`
   helper raises cleanly when orientation metadata is required but ffprobe
   is absent, and `VideoReader` works without rotation for fixtures that
   don't have any.
4. **Large mediapipe install at Stage 1.** Already addressed by ADR 001
   (`[pipeline]` extra) and this stage's first task pre-installs it.

## Out of scope (explicitly)

- Club-head detection (Stage 2).
- Phase segmentation (Stage 4).
- Any comparison / feedback (Stages 5–6).
- Gradio demo (Stage 8).
- Real GolfDB data (Stage 3).
- Training-free heuristic pose fallbacks; MediaPipe is the only backend in
  V1.

## Exit criterion (from `CLAUDE.md` §5.1)

`swingscan pose --input tests/fixtures/sample_swing.mp4 --output /tmp/out.parquet`
produces a parquet file, and a separate helper reloads it into a
`PoseSequence` whose length matches the original video's frame count.

## Execution order

1. Pre-install `[pipeline]` extras and ffmpeg (already launched).
2. Write `utils/ffmpeg.py`, `io/video.py` + tests.
3. Generate synthetic fixture video.
4. Write `pose/keypoints.py`, `pose/base.py` + tests.
5. Write `pose/mediapipe_backend.py` + smoke test.
6. Write `io/serialize.py` + round-trip test.
7. Wire the `pose` CLI subcommand.
8. Run lint / typecheck / test. Run the full CLI on the fixture. Write
   `docs/plans/stage1-status.md`. Commit. Merge to `main`.
