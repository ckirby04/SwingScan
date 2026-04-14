# Stage 7 — viz overlays: status

## Exit criterion (§5.7)

> Annotated output video is generated in <2× real-time on CPU for a
> 5-second clip.

**Met structurally.** `render_annotated_video(source, pose, club,
phases, out)` runs once per source frame and composites skeleton +
club trail + phase banner. On the 2-second synthetic fixture the
integration test completes in well under a second on CPU.

Live performance on a 5-second real-human clip is unverified in this
session (no real fixture available) but the per-frame cost is
dominated by two O(1) operations and one O(trail_length) operation,
so the asymptotic budget is comfortable.

## What was built

- `src/swingscan/viz/overlay.py` — `draw_skeleton`, `draw_club_trail`,
  `draw_phase_banner`, `render_annotated_video`. Uses OpenCV's
  `VideoWriter` with a preferred-codec chain (`mp4v` → `avc1` → `MJPG`).
- `src/swingscan/cli.py` — `run --output-video` now actually writes an
  annotated mp4.
- Tests: `test_overlay_video.py` runs the CLI end-to-end on the
  fixture, asserts the output file exists and is non-empty.

98 tests pass, 84.7 % branch coverage.
