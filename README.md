# SwingScan

**Video in, coaching out.** SwingScan takes a single amateur golf-swing
clip, extracts 33 body keypoints (MediaPipe BlazePose), segments the
swing into the 8 canonical events (a vendored SwingNet model from the
[GolfDB](https://github.com/wmcnally/golfdb) CVPR 2019 paper), compares
your pose at each event against a bank of pro swings, and returns a
short list of plain-English coaching cues — the kind of thing a teaching
pro would actually say to a double-digit handicapper.

It runs entirely offline on a laptop CPU. No cloud, no uploads, no
analytics.

> SwingScan is a coaching-support tool. Every cue it emits is a
> heuristic, not a medical or clinical assessment.

## What the output looks like

On a real GolfDB face-on driver clip, the pipeline's text report is
something like:

```
SwingScan report: 128 frames
  club: 0% missing, 0% interpolated
  phases: ADDRESS=1, TOE_UP=18, MID_BACKSWING=28, TOP=37,
          MID_DOWNSWING=42, IMPACT=45, MID_FOLLOW_THROUGH=48, FINISH=64
  cohort: 178 pro swings

SwingScan feedback (heuristic coaching cues):
  [!!!] IMPACT · You're standing up through impact — spine straightening
        about 13° more than pros. This is the #1 cause of fat, thin, and
        shanked shots for amateurs. Feel: keep your trail cheek pointed
        at the ground and let your hips clear BEHIND you, not toward the
        ball. Drill: set up with a chair cushion against your trail hip.
        Don't let your hip push into it in the downswing.
```

The same pipeline can also render an annotated mp4 overlay — skeleton,
phase banner, and (optionally) a club-head trail.

## How it works

1. **Video I/O.** `swingscan.io.video.VideoReader` wraps OpenCV with
   ffmpeg, handles rotation metadata, and yields BGR frames.
2. **Pose.** `MediaPipePoseEstimator` runs BlazePose on every frame and
   emits `PoseFrame` records (33 joints × image + world coords, plus a
   low-confidence flag).
3. **Phase segmentation.** `SwingNetSegmenter` (the vendored upstream
   model, CC BY-NC 4.0) picks the 8 event frames — Address / Toe-up /
   Mid-backswing / Top / Mid-downswing / Impact / Mid-follow-through /
   Finish. Without weights, it falls back to a wrist-velocity heuristic.
4. **Club head.** `HeuristicClubDetector` uses MediaPipe's hand finger
   landmarks to estimate shaft direction at the grip, extrapolated by
   forearm length. `ClubTracker` smooths and gap-fills the result.
5. **Metrics.** `compute_swing_metrics` reads the pose at each of the
   8 events and computes hip rotation, shoulder rotation, x-factor,
   spine lean, lead arm straightness, wrist hinge, lead knee flex, and
   head drift.
6. **Cohort comparison.** `ProBank` stores a per-event pose snapshot
   per pro swing. `compare_against_bank` produces circular-aware
   z-scores of the amateur's metrics against the cohort.
7. **Feedback.** `RuleEngine` loads
   [`configs/feedback_rules.yaml`](configs/feedback_rules.yaml) — 15
   rules calibrated against the real pro bank distribution — and emits
   severity-ordered coaching cues.
8. **Visualization.** `render_annotated_video` writes an mp4 with the
   pose skeleton, phase banner, and (opt-in) club trail.

Details live in
[`docs/architecture.md`](docs/architecture.md),
[`docs/pipeline.md`](docs/pipeline.md), and the
[`docs/plans/`](docs/plans/) per-stage status docs.

## Measured quality

Ad-hoc evaluation on 50 face-on driver swings from GolfDB against the
full 178-swing face-on driver pro bank:

| Segmenter                | PCE (±5 frames) | PCE (±1 frame) |
|--------------------------|:---------------:|:--------------:|
| Wrist-velocity heuristic |      9.5 %      |        —       |
| **SwingNet (vendored)**  |   **94.75 %**   |   **82.5 %**   |

Pose completeness: 99.6 %. Pro-bank coverage: 100 %. Full reports under
[`docs/evaluations/`](docs/evaluations/). The SwingNet tolerance=1
number is inflated relative to the paper's 71.5 % — our eval subset
overlaps with SwingNet's training split. Honest held-out numbers are
V1.1 work (see [`ROADMAP.md`](ROADMAP.md)).

## Quickstart

### Requirements

- **Python 3.11** (pinned via `.python-version` and `pyproject.toml`).
- **ffmpeg** on PATH for video decoding.
- **GNU Make** for the developer targets below — optional if you'd
  rather run the underlying Python commands directly.

On Windows:

```powershell
winget install Python.Python.3.11
winget install Gyan.FFmpeg
winget install GnuWin32.Make    # optional
```

### Install

```bash
make install     # creates .venv and installs dev extras
```

From there, install the heavy pipeline extras (torch, mediapipe,
opencv, ultralytics, gradio) that the real pipeline needs:

```bash
.venv/bin/python -m pip install -e ".[dev,pipeline]"      # POSIX
.venv\Scripts\python.exe -m pip install -e ".[dev,pipeline]"  # Windows
```

### Verify

```bash
make lint
make typecheck
make test
make version     # prints "swingscan 0.1.0"
```

### Run the CLI

```bash
swingscan pose   --input swing.mp4 --output pose.parquet
swingscan phases --input swing.mp4 --output phases.json
swingscan run    --input swing.mp4 \
                 --output report.json \
                 --output-video annotated.mp4 \
                 --pro-bank data/pro_bank/bank.parquet
```

`swingscan run` auto-discovers `models/swingnet_1800.pth.tar` and
`data/pro_bank/bank.parquet` if they exist, so the flags above are only
needed when you want to override the default paths.

### Run the local demo

```bash
make demo
# or, equivalently:
python scripts/demo_local.py
```

Opens a Gradio app at `http://127.0.0.1:7860`. Drop a face-on driver
swing video (≤30 s, ≤100 MB) into the uploader; the app returns an
annotated video, a per-phase metrics table, and the coaching-cue list.

### Build the pro reference bank from GolfDB

```bash
# One-time data setup
python scripts/download_golfdb.py               # 692 KB annotation pickle
python -m gdown "https://drive.google.com/uc?id=1uBwRxFxW04EqG87VCoX3l6vXeV5T5JYJ" \
    -O data/raw/golfdb/videos_160.zip           # 699 MB, CC BY-NC 4.0
python -c "import zipfile; zipfile.ZipFile('data/raw/golfdb/videos_160.zip').extractall('data/raw/golfdb/')"
python -m gdown "https://drive.google.com/uc?id=1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW" \
    -O models/swingnet_1800.pth.tar             # 63 MB SwingNet weights

# Convert annotations → labels.json → pro bank parquet
python scripts/convert_golfdb_labels.py
python scripts/build_pro_bank.py
```

The GolfDB artifacts are distributed by the upstream authors under
CC BY-NC 4.0 for non-commercial research use.

## Repository layout

| Path | Purpose |
|---|---|
| `src/swingscan/` | The importable package (pose, club, phases, metrics, compare, feedback, viz, pipeline, CLI). |
| `scripts/` | Data + operations CLIs: download, convert, build, evaluate, calibrate, demo. |
| `configs/` | `default.yaml` runtime config plus the `feedback_rules.yaml` rule set. |
| `tests/` | 116 unit + integration tests, all run in CI. |
| `docs/` | `architecture.md`, `pipeline.md`, per-stage status docs, ADRs, evaluation reports. |
| `models/` | User-local model weights. Gitignored. |
| `data/` | User-local datasets and derived artifacts. Gitignored. |
| `.github/workflows/ci.yml` | GitHub Actions: ruff, ruff format check, mypy strict, pytest. |

## Scope guardrails (V1)

SwingScan V1 is explicitly **not**:

- a launch monitor, ball-flight simulator, or shot-shape predictor,
- a multi-person or crowd analysis tool,
- a mobile app,
- a clinical or medical device.

See [`CLAUDE.md`](CLAUDE.md) §2 for the full scope statement and
operating principles.

## Licensing and attribution

- SwingScan is MIT licensed (see `pyproject.toml`).
- **MediaPipe** is Apache 2.0 (Google).
- **SwingNet** (McNally et al., CVPR 2019 Workshops) is distributed
  under CC BY-NC 4.0 via the upstream
  [`wmcnally/golfdb`](https://github.com/wmcnally/golfdb) repo. The
  model architecture is vendored at
  `src/swingscan/phases/_swingnet_model.py` with attribution in the
  module header. If you use SwingScan please also cite the GolfDB
  paper.
- **GolfDB** videos and annotations are CC BY-NC 4.0. SwingScan
  never commits their content to git; see `data/README.md` for the
  download flow.

## Where next

See [`ROADMAP.md`](ROADMAP.md). The biggest open items:

- Train a real YOLO club-head detector (the current heuristic is
  why the club trail is hidden by default — pass `--draw-club` to
  see it).
- Handedness and view auto-detection at upload time.
- Honest held-out SwingNet eval on GolfDB split 4 only.
- Dependency lock file so `pip install -e .[dev,pipeline]` is
  byte-reproducible.

Contributions welcome — run `make lint typecheck test` before opening
a PR; CI will run the same checks on push.
