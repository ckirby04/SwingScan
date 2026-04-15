# SwingScan

SwingScan is a computer-vision pipeline that ingests a single amateur golf
swing video, extracts body keypoints and club-head trajectory, segments the
swing into the 8 canonical phases (Address → Finish), compares it against a
bank of professional reference swings, and returns prioritized biomechanical
feedback tied to specific phases and joints. It is a coaching-support tool,
not a clinical or medical device.

The full design, stage plan, and operating principles live in
[`CLAUDE.md`](./CLAUDE.md) — read that file before making non-trivial
changes.

## Status

SwingScan is built in strict stage gates. Each stage merges only after its
exit criterion is met. The active branch is the one currently iterating on
the in-progress stage.

| Stage | Name                              | Status  | Notes |
|-------|-----------------------------------|---------|-------|
| 0     | Bootstrap                         | Done    | Scaffold, tooling, smoke test. [status](docs/plans/stage0-status.md) |
| 1     | Video I/O + Pose (MediaPipe)      | Done    | `swingscan pose` CLI, parquet schema. [status](docs/plans/stage1-status.md) |
| 2     | Club-head detection               | Done    | Heuristic + YOLO, `ClubTracker`. [status](docs/plans/stage2-status.md) · [ADR 002](docs/decisions/002-club-detection-fallback.md) |
| 3     | GolfDB ingestion + pro bank       | Done    | `ProBank`, download + build scripts. [status](docs/plans/stage3-status.md) |
| 4     | Phase segmentation                | Done    | Heuristic + **real SwingNet inference** (94.75 % PCE on 50-swing eval). [status](docs/plans/stage4-status.md) · [eval report](docs/evaluations/) |
| 5     | Biomech metrics + comparison      | Done    | `SwingMetrics`, `SwingDiff`. [status](docs/plans/stage5-status.md) |
| 6     | Feedback generation               | Done    | 15-rule YAML engine, text + JSON. [status](docs/plans/stage6-status.md) |
| 7     | Visualization overlays            | Done    | Annotated video output. [status](docs/plans/stage7-status.md) |
| 8     | Demo UI (Gradio)                  | Done    | `make demo`. [status](docs/plans/stage8-status.md) |
| 9     | Evaluation harness                | Done    | `scripts/evaluate_pipeline.py`. [status](docs/plans/stage9-status.md) |
| 10    | Polish + handoff                  | Done    | v0.1.0 tag, ROADMAP, architecture docs. |

**Post-v0.1.0 work landed on main:**
- SwingNet inference wired up — 9.5 % PCE → 94.75 % PCE on the same
  50-swing face-on driver eval.
- Feedback thresholds recalibrated from the real 178-swing bank via
  `scripts/calibrate_thresholds.py`. 4-of-5 false-positive feedback
  cues on Jerry Kelly's swing went away.
- Circular-mean statistics for angular metrics (fixes 127° shoulder
  rotation wraparound noise).
- UTF-8 stdout in the CLI so Windows PowerShell renders °, —, and ·.
- GitHub Actions CI running lint / ruff format / mypy / pytest on
  every push and PR against `main`.
- A real data/pro_bank/bank.parquet (178 face-on driver pros) and a
  data/pro_bank_dtl/ down-the-line bank built from the same release.

## Quickstart

### Requirements

- **Python 3.11** (pinned — see `.python-version` and `pyproject.toml`).
- **GNU Make** for the developer targets below. On Windows, install via
  `winget install GnuWin32.Make` or run the underlying commands directly.
- **ffmpeg** is required starting in Stage 1 for robust video decoding; not
  required for Stage 0.

### Install

```bash
# Create a venv and install Stage 0 dev dependencies.
make install

# Run linters, type checker, and tests.
make lint
make typecheck
make test

# Confirm the CLI entry point works.
make version
```

From Stage 1 onward you will also need the heavy pipeline extras (`torch`,
`mediapipe`, `opencv-python`, `ultralytics`, `gradio`). Install them with:

```bash
.venv/bin/python -m pip install -e ".[dev,pipeline]"   # POSIX
.venv\Scripts\python.exe -m pip install -e ".[dev,pipeline]"  # Windows
```

See [ADR 001](docs/decisions/001-dependency-extras.md) for why heavy deps
are split out.

### Run the pipeline

```bash
# Pose extraction only (parquet output).
swingscan pose --input tests/fixtures/sample_swing.mp4 --output pose.parquet

# Phase segmentation only.
swingscan phases --input tests/fixtures/sample_swing.mp4 --output phases.json

# Full run: pose + club + phases + optional feedback + annotated video.
swingscan run \
  --input tests/fixtures/sample_swing.mp4 \
  --output report.json \
  --output-video annotated.mp4 \
  --pro-bank data/pro_bank/bank.parquet  # optional; enables feedback
```

### Launch the local demo

```bash
make demo
# → http://127.0.0.1:7860
```

The demo is local-only, binds to 127.0.0.1 by default, caps uploads at
100 MB / 30 s, and does not persist input videos.

## Scope guardrails (V1)

SwingScan V1 is **not**:

- a launch monitor, ball-flight simulator, or shot-shape predictor,
- a multi-person or crowd analysis tool,
- a mobile application,
- a clinical or medical device.

All feedback is coaching guidance only. See `CLAUDE.md` §2 for the full
scope statement.

## Licensing

SwingScan itself is MIT licensed. It depends on third-party projects with
their own licenses and citation requirements — notably MediaPipe
(Apache-2.0) and GolfDB / SwingNet (see the GolfDB CVPR 2019 paper). A
`THIRD_PARTY.md` will be added as the relevant stages land.

**Do not commit any video, dataset file, or model weight.** See
`data/README.md` and `models/README.md`.

## Project layout

See `CLAUDE.md` §4 for the canonical tree. The shipped package lives under
`src/swingscan/`; tests under `tests/`; executable scripts (dataset
download, evaluation, demo) under `scripts/`. Documentation — including
stage plans, stage status docs, and architecture decision records — lives
under `docs/`.
