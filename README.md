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

| Stage | Name                              | Status        | Notes |
|-------|-----------------------------------|---------------|-------|
| 0     | Bootstrap                         | In progress   | Scaffold, tooling, smoke test. Branch `stage-0-bootstrap`. Status: [`docs/plans/stage0-status.md`](docs/plans/stage0-status.md). |
| 1     | Video I/O + Pose (MediaPipe)      | Not started   | — |
| 2     | Club-head detection               | Not started   | — |
| 3     | GolfDB ingestion + pro bank       | Not started   | — |
| 4     | Phase segmentation (SwingNet)     | Not started   | — |
| 5     | Biomech metrics + comparison      | Not started   | — |
| 6     | Feedback generation               | Not started   | — |
| 7     | Visualization overlays            | Not started   | — |
| 8     | Demo UI (Gradio)                  | Not started   | — |
| 9     | Evaluation harness                | Not started   | — |
| 10    | Polish + handoff                  | Not started   | — |

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

### Run the pipeline (not yet available)

Will be wired up in Stage 1:

```bash
swingscan pose --input tests/fixtures/sample_swing.mp4 --output /tmp/pose.parquet
swingscan phases --input tests/fixtures/sample_swing.mp4 --output /tmp/phases.json
swingscan run --input tests/fixtures/sample_swing.mp4 --output-video /tmp/out.mp4
```

Today (Stage 0) only `swingscan version` and a `swingscan run` stub exist.

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
