# SwingScan — Golf Swing Analysis System

You are the lead engineer on **SwingScan**, a computer vision system that ingests an amateur golf swing video, extracts body + club keypoints, segments the swing into canonical phases, compares it against a database of professional swings, and returns targeted biomechanical feedback ("rotate hips 12° more at P4", "lead arm collapsing at P6", etc.).

This document is your operating manual. Read it fully before taking any action. You are being dropped into an **empty directory** and are expected to bootstrap the entire project from scratch, in stages, verifying each stage before moving to the next.

---

## 1. Operating Principles

1. **Stage gates are mandatory.** Do not skip ahead. Each stage has an exit criterion. If the criterion is not met, stop and report rather than improvising forward.
2. **Work on a branch, commit often.** Every stage gets its own feature branch off `main`. Every meaningful unit of work gets a commit with a conventional-commits-style message (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`). Never commit to `main` directly after the initial scaffold.
3. **Plan before coding.** For any stage beyond Stage 0, write a short plan to `docs/plans/stageN-plan.md` before writing code. The plan lists: files to be created/modified, functions/classes with signatures, test cases, and risks. Then execute against the plan.
4. **Test what you write.** Every non-trivial module gets unit tests. Use `pytest`. Aim for tests that actually exercise the logic, not placeholder asserts. Target ≥70% coverage on core modules (pose, phases, metrics, feedback).
5. **Fail loudly.** No silent `except:` blocks. No `print` for errors — use the `logging` module configured in `swingscan/utils/logging.py`.
6. **Deterministic where possible.** Set seeds for numpy, torch, and random in any training or evaluation script. Pin dependency versions in `pyproject.toml`.
7. **Document as you go.** Every module gets a docstring at the top explaining its role in the pipeline. Every public function gets a docstring with Args/Returns/Raises.
8. **Never fabricate data or results.** If a dataset download fails, a pretrained weight is unavailable, or a metric can't be computed, say so explicitly and surface the blocker. Do not generate synthetic "example output" that could be mistaken for real evaluation results.
9. **Ask before destructive actions.** Deleting data, force-pushing, rewriting history, or wiping model checkpoints requires an explicit user request.
10. **When in doubt, write a decision record.** Non-obvious architectural choices go in `docs/decisions/NNN-title.md` as lightweight ADRs (context, decision, consequences).

---

## 2. Project Vision & Scope

**What it is:** A pipeline + CLI + lightweight web demo that takes a single-swing video (face-on or down-the-line view) and returns:
- Per-frame body keypoints (33 joints via MediaPipe BlazePose baseline, upgradeable to MMPose).
- Club head trajectory (via a fine-tuned small object detector).
- Segmentation into the 8 canonical swing events (Address, Toe-up, Mid-backswing, Top, Mid-downswing, Impact, Mid-follow-through, Finish) using SwingNet as the baseline.
- A comparison against a pro reference bank derived from GolfDB.
- A ranked list of biomechanical feedback items, each tied to (phase, joint, magnitude, direction).

**What it is not** (V1 scope discipline):
- Not a ball-flight simulator, launch monitor replacement, or shot-shape predictor.
- Not a multi-person / crowd analysis tool. Assume one golfer, mostly centered, mostly static camera.
- Not a mobile app yet. V1 runs on a workstation with a Python CLI + a minimal FastAPI/Gradio demo.
- Not a clinical or medical device. All feedback is coaching guidance, never health advice.

**Success definition for V1:**
- Given a clean face-on driver swing video, the system returns pose + phases + at least 3 actionable feedback items in under 60 seconds on a single consumer GPU.
- The pro reference bank covers at least 100 distinct GolfDB swings across face-on and down-the-line views.
- End-to-end pipeline is runnable via a single CLI command on a fresh checkout after `make install`.

---

## 3. Tech Stack (pinned)

- **Language:** Python 3.11
- **Env/packaging:** `uv` if available, else `pip` + `venv`. `pyproject.toml` is the source of truth.
- **Core libs:** `numpy`, `scipy`, `opencv-python`, `mediapipe`, `torch`, `torchvision`, `ultralytics` (YOLO), `pandas`, `scikit-learn`.
- **Video I/O:** `opencv-python` for frames, `ffmpeg` (system) for transcoding. Never assume ffmpeg is installed — check and report if missing.
- **Serving (V1 demo):** `gradio` (preferred for speed) or `fastapi` + a tiny static page. Pick one in Stage 6 and commit.
- **Testing:** `pytest`, `pytest-cov`.
- **Linting/formatting:** `ruff` (format + lint). No `black`, no `isort` — `ruff` does both.
- **Type checking:** `mypy` in strict mode on the `swingscan/` package, lenient on scripts and tests.
- **Config:** `pydantic` v2 models for all configs. YAML files under `configs/` are loaded into these models.
- **Logging:** stdlib `logging`, configured once in `swingscan/utils/logging.py`.

Do not introduce new dependencies without adding them to `pyproject.toml` and explaining the choice in the stage plan.

---

## 4. Repository Layout (target)

Create this structure during Stage 0. Do not deviate without a decision record.

```
swingscan/
├── README.md
├── CLAUDE.md                    # this file, copied in at bootstrap
├── pyproject.toml
├── Makefile
├── .gitignore
├── .python-version
├── .pre-commit-config.yaml
├── configs/
│   ├── default.yaml
│   ├── pose_mediapipe.yaml
│   ├── phases_swingnet.yaml
│   └── feedback_rules.yaml
├── data/                        # gitignored, populated by scripts
│   ├── raw/
│   ├── processed/
│   ├── pro_bank/
│   └── README.md                # explains where data comes from
├── models/                      # gitignored, populated by scripts
│   └── README.md
├── docs/
│   ├── architecture.md
│   ├── pipeline.md
│   ├── plans/
│   └── decisions/
├── notebooks/                   # exploratory only, not for production code
├── scripts/
│   ├── download_golfdb.py
│   ├── build_pro_bank.py
│   ├── evaluate_pipeline.py
│   └── demo_local.py
├── src/
│   └── swingscan/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── pipeline.py          # top-level orchestration
│       ├── io/
│       │   ├── __init__.py
│       │   ├── video.py
│       │   └── serialize.py
│       ├── pose/
│       │   ├── __init__.py
│       │   ├── base.py          # PoseEstimator protocol
│       │   ├── mediapipe_backend.py
│       │   └── keypoints.py     # joint index enum + helpers
│       ├── club/
│       │   ├── __init__.py
│       │   ├── detector.py      # YOLO wrapper
│       │   └── tracker.py       # temporal smoothing
│       ├── phases/
│       │   ├── __init__.py
│       │   ├── events.py        # 8-event enum
│       │   ├── swingnet.py      # wraps GolfDB SwingNet
│       │   └── segmenter.py
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── angles.py        # joint-angle computations
│       │   ├── normalize.py     # scale / orientation normalization
│       │   └── biomech.py       # hip turn, shoulder turn, X-factor, etc.
│       ├── compare/
│       │   ├── __init__.py
│       │   ├── pro_bank.py      # load/query pro reference trajectories
│       │   └── diff.py          # per-phase comparison
│       ├── feedback/
│       │   ├── __init__.py
│       │   ├── rules.py         # symbolic rule engine
│       │   └── render.py        # human-readable messages
│       ├── viz/
│       │   ├── __init__.py
│       │   └── overlay.py       # draw keypoints + club on frames
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           ├── seeding.py
│           └── paths.py
└── tests/
    ├── conftest.py
    ├── fixtures/                # tiny sample video, synthetic pose sequences
    ├── unit/
    └── integration/
```

---

## 5. Stage Plan

Each stage ends with a commit on its own branch and a written status update in `docs/plans/stageN-status.md`. Merge to `main` only after the exit criterion is met.

### Stage 0 — Bootstrap
**Goal:** Repo exists, runs, passes lint, passes a trivial test.

Tasks:
1. `git init`, create `main` branch.
2. Create the directory tree from §4. Use empty `.gitkeep` files where needed.
3. Write `pyproject.toml` with pinned deps (see §3). Include `[project.scripts]` entry `swingscan = "swingscan.cli:main"`.
4. Write `Makefile` with targets: `install`, `lint`, `format`, `test`, `typecheck`, `demo`, `clean`.
5. Write `.gitignore` covering `data/`, `models/`, `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `*.mp4` except `tests/fixtures/*.mp4`.
6. Write `README.md` with: one-paragraph pitch, install, quickstart, status table pointing to each stage's status doc.
7. Write `src/swingscan/utils/logging.py`, `src/swingscan/utils/seeding.py`, `src/swingscan/config.py`.
8. Write a stub `cli.py` with a `version` subcommand and a `run` subcommand (no-op stub).
9. Write `tests/unit/test_smoke.py` that imports the package and asserts version is set.
10. Configure `ruff`, `mypy`, `pytest` in `pyproject.toml`.
11. Install pre-commit hooks for `ruff` and basic checks.

**Exit criterion:** `make install && make lint && make typecheck && make test` all pass on a fresh checkout. `swingscan version` prints a version.

---

### Stage 1 — Video I/O and Pose Extraction (MediaPipe baseline)
**Goal:** Given an input video, output a JSON/parquet of per-frame 33-joint keypoints.

Tasks:
1. `io/video.py`: `VideoReader` class (frame iterator, fps, dims, duration), handling rotation metadata. Test against a short fixture video.
2. `pose/keypoints.py`: `Joint` enum with names matching BlazePose's 33 landmarks. Helper groupings: `HEAD`, `SHOULDERS`, `ELBOWS`, `WRISTS`, `HIPS`, `KNEES`, `ANKLES`, `FEET`.
3. `pose/base.py`: `PoseEstimator` `typing.Protocol` — `estimate(frame) -> PoseFrame`, `estimate_video(video) -> PoseSequence`.
4. `pose/mediapipe_backend.py`: concrete implementation wrapping `mediapipe.solutions.pose`. Returns normalized (image-space) AND world coordinates when available. Flag low-confidence frames.
5. `io/serialize.py`: save/load a `PoseSequence` to parquet and JSON. Schema documented in `docs/pipeline.md`.
6. CLI: `swingscan pose --input video.mp4 --output pose.parquet`.
7. Tests: feed a 2–3 second fixture clip (commit a permissively-licensed one or generate a synthetic moving-shape video), assert frame count, assert non-empty keypoints on at least 80% of frames, assert serialization round-trips.

**Exit criterion:** `swingscan pose --input tests/fixtures/sample_swing.mp4 --output /tmp/out.parquet` produces a file, and a separate script can load it and render overlays matching the original video length.

---

### Stage 2 — Club Head Detection
**Goal:** Per-frame club head position (x, y, confidence), temporally smoothed.

Context: Body pose models do NOT track the club head. You need a separate small object detector. The GolfDB bounding boxes include club+ball through the swing, which gives you a starting label set if you can re-download source videos — but do not block Stage 2 on that. Begin with a YOLOv8 nano model fine-tuned on a small self-labeled dataset, or as a fallback, use classical CV (motion + color + line detection near the hands) and flag it as a placeholder backend.

Tasks:
1. Write `scripts/label_club.py`: a tiny CLI that opens a video frame-by-frame and lets a user click the club head, saving YOLO-format labels. (Not required to run — just exists for future expansion.)
2. `club/detector.py`: `ClubDetector` with two backends:
   - `YoloClubDetector` (requires a trained weight file; loads from `models/club_yolo.pt` if present).
   - `HeuristicClubDetector` (hand-to-club geometric extrapolation from pose — uses wrist positions + estimated club length — as a fallback when the YOLO weights are absent).
3. `club/tracker.py`: Kalman filter or simple exponential smoothing over raw detections, interpolating gaps ≤ 3 frames.
4. Extend pipeline so `swingscan run` produces combined pose + club trajectory.
5. Tests: synthetic fixture where a "club head" ground truth is known; assert tracked trajectory stays within a tolerance.

**Exit criterion:** Running the pipeline on the fixture produces a club trajectory with no gaps, and the heuristic backend runs without needing downloaded model weights.

**Decision record required:** Write an ADR on how you chose the fallback strategy and what "good enough" means for club tracking in V1.

---

### Stage 3 — GolfDB Ingestion and Pro Reference Bank
**Goal:** Build a local database of pro swing trajectories to compare against.

Important constraints — read carefully:
- GolfDB consists of YouTube URLs and preprocessed 160×160 clips. Some source videos may no longer be available; handle failures gracefully.
- Do not commit any video files. Do not commit any YouTube-downloaded content. Everything goes under `data/raw/` which is gitignored.
- The 160×160 clips are too small for quality pose extraction — you need either the original YouTube videos or you need to be honest about the resolution limitation.
- Respect YouTube terms of service. Use `yt-dlp` only if the user has it installed and explicitly opts in via a config flag; otherwise fall back to the 160×160 clips and document the quality tradeoff.

Tasks:
1. `scripts/download_golfdb.py`: clones/fetches the GolfDB annotation CSV from the public repo, saves to `data/raw/golfdb/`. Does NOT auto-download videos. Prints instructions for the two download modes (low-res clips vs. yt-dlp).
2. Implement the low-res-clip path first as the default. Write clearly in `data/README.md` that V1 pro bank quality is bounded by input resolution.
3. `scripts/build_pro_bank.py`: iterates over available swings, runs the pose pipeline, segments into phases (Stage 4 will provide this — for now, use the GolfDB event frame labels directly since they already mark the 8 events), computes normalized joint-angle time series per phase, saves to `data/pro_bank/bank.parquet`.
4. `compare/pro_bank.py`: `ProBank` class — load, filter by (view type, handedness, club type), query by phase.
5. Tests: a synthetic mini-bank of 3 fake swings; verify load, filter, query correctness.

**Exit criterion:** `python scripts/build_pro_bank.py --limit 20` runs end-to-end on whatever GolfDB data is available, producing a queryable bank file. Count of successfully processed swings is logged and written to `data/pro_bank/manifest.json`.

---

### Stage 4 — Swing Phase Segmentation
**Goal:** Given a full swing video, identify frame indices for the 8 canonical events.

Tasks:
1. `phases/events.py`: `SwingEvent` enum (`ADDRESS`, `TOE_UP`, `MID_BACKSWING`, `TOP`, `MID_DOWNSWING`, `IMPACT`, `MID_FOLLOW_THROUGH`, `FINISH`).
2. `phases/swingnet.py`: wrap the SwingNet baseline from the GolfDB repo. Provide a `load_swingnet(weights_path)` helper and an `infer(video_tensor) -> list[int]` function. If weights are absent, raise a clear error pointing at `models/README.md` with download instructions.
3. `phases/segmenter.py`: high-level `PhaseSegmenter` that takes a `VideoReader` and returns a `dict[SwingEvent, int]`. Include a secondary pose-based heuristic segmenter (`HeuristicSegmenter`) that uses wrist velocity + hand position zero-crossings to estimate events — this is the fallback when SwingNet weights aren't available.
4. CLI: `swingscan phases --input video.mp4 --output phases.json`.
5. Tests: a hand-labeled mini fixture; assert each detected event frame is within ±5 frames of ground truth for the heuristic segmenter (loose tolerance is fine for V1).

**Exit criterion:** Both backends run. The heuristic one works with no external weights. Running `swingscan phases` on the sample fixture produces all 8 events in monotonic order.

---

### Stage 5 — Biomechanical Metrics and Comparison
**Goal:** Compute per-phase metrics and diff against the pro bank.

Tasks:
1. `metrics/normalize.py`: scale by shoulder width (face-on) or torso length (down-the-line), align by hip center, handle left-handed flip.
2. `metrics/angles.py`: functions for each metric — `hip_rotation(pose, view)`, `shoulder_rotation(pose, view)`, `x_factor(pose)`, `spine_angle(pose, view)`, `lead_arm_straightness(pose)`, `wrist_hinge(pose)`, `knee_flex(pose)`, `weight_shift_proxy(pose)`, `head_movement(pose, reference_frame)`. Each returns a scalar or a named tuple with units.
3. `metrics/biomech.py`: composite metrics computed across phases — e.g., `backswing_tempo = (top_frame - address_frame) / (impact_frame - top_frame)`.
4. `compare/diff.py`: `SwingDiff` dataclass with per-phase, per-metric deltas vs. a matched cohort from the pro bank (match on view, handedness, club). Expose magnitude, sign, and a z-score against the cohort distribution.
5. Tests: unit tests on synthetic pose sequences with known angles (e.g., an arm rotated exactly 90° returns ~90°).

**Exit criterion:** Given a pose sequence + phase map + pro bank, the pipeline produces a `SwingDiff` with at least 8 computed metrics per phase and z-scores.

---

### Stage 6 — Feedback Generation
**Goal:** Turn a `SwingDiff` into prioritized, human-readable coaching cues.

Tasks:
1. `configs/feedback_rules.yaml`: a rule set — each rule has (name, applies_to_phase, metric, threshold, direction, message_template, severity_weight). Start with ~15 rules covering the common amateur faults (early extension, over-the-top, casting, reverse pivot, sway, lead arm breakdown, open/closed clubface proxy, head dip, hip stall, insufficient turn).
2. `feedback/rules.py`: `RuleEngine` loads the YAML into typed `FeedbackRule` objects and evaluates them against a `SwingDiff`, returning a list of `FeedbackItem` (message, phase, metric, severity).
3. `feedback/render.py`: sorts by severity, caps at N items (default 5), and renders plain text + optional JSON payload. Each item must cite the phase and the measured delta so the user can trust it.
4. Be explicit in docstrings that these are heuristic cues, not coaching certifications.
5. Tests: fabricate a `SwingDiff` with known deltas, assert specific rules fire.

**Exit criterion:** `swingscan run --input video.mp4` prints a full report: detected phases, key metrics, top feedback items.

---

### Stage 7 — Visualization Overlays
**Goal:** Produce an annotated output video so the user can see what the system sees.

Tasks:
1. `viz/overlay.py`: `draw_skeleton`, `draw_club_trail`, `draw_phase_banner` (show current phase label), `draw_metric_panel` (small HUD showing live angles).
2. Extend CLI: `swingscan run --input v.mp4 --output-video out.mp4`.
3. Handle FPS and codec correctly. Prefer `mp4v` or `avc1` fallback chain.
4. Tests: smoke test that the output file exists, has ≥1 frame, and has expected duration within tolerance.

**Exit criterion:** Annotated output video is generated in <2× real-time on CPU for a 5-second clip.

---

### Stage 8 — Demo UI
**Goal:** A minimal web demo for dropping in a video and getting feedback.

Tasks:
1. Pick **Gradio** for V1 (ADR required if choosing otherwise).
2. `scripts/demo_local.py`: Gradio app with a video upload widget, runs the pipeline, and returns (annotated video, metrics table, feedback list).
3. Make it launchable via `make demo`.
4. Hard cap upload size and clip duration; reject anything >30 seconds or >100 MB with a clear message.
5. No authentication, no persistence, no analytics — this is a local demo only.

**Exit criterion:** `make demo` opens a browser, user can upload a sample video, and a report is returned end-to-end.

---

### Stage 9 — Evaluation Harness
**Goal:** You can measure whether changes to the pipeline make it better or worse.

Tasks:
1. `scripts/evaluate_pipeline.py`: runs the full pipeline on a held-out subset of GolfDB, compares detected events to ground truth (Percentage of Correctly detected Events, PCE, as defined in the GolfDB paper), and writes a report.
2. Add a sanity check on pose completeness (% frames with valid keypoints).
3. Add a report on pro bank coverage (how many pro swings matched a typical amateur input on view/handedness/club).
4. Tests: evaluate script runs on a fixture bank and produces a report file.

**Exit criterion:** A single command produces a reproducible evaluation report with numbers, not placeholders.

---

### Stage 10 — Polish and Handoff
- Fill in `docs/architecture.md` and `docs/pipeline.md` based on what was actually built.
- Ensure README quickstart works on a clean machine.
- Tag `v0.1.0`.
- Write a `ROADMAP.md` listing V2 ideas: learned swing embeddings (à la GolfMate), 3D lifting (VideoPose3D / MotionBERT), multi-view fusion, mobile deployment, custom club-head detector training on labeled GolfDB frames.

---

## 6. Data and Licensing Guardrails

- **Do not commit any video, pretrained weight, or dataset file.** Everything lives under gitignored `data/` and `models/`.
- **Do not scrape or redistribute GolfDB videos.** Reference the public repo and let the user run the download script at their own discretion.
- **Honor licenses.** If you use SwingNet weights, respect the GolfDB repo's license and cite the CVPR 2019 paper in the README. If you use MediaPipe, cite Google's Apache-2.0 license in `THIRD_PARTY.md`.
- **No real golfer likenesses in committed fixtures.** Test fixtures must be either synthetic, self-recorded, or under a clearly permissive license. If unsure, use synthetic.
- **No medical or clinical framing in any user-visible string.** Feedback is coaching guidance only.

---

## 7. Security and Privacy

- The demo accepts user-uploaded video. Do not persist uploads by default. Log only filenames and durations, never file contents.
- No network calls from the core pipeline. Downloads happen only in explicitly-invoked `scripts/` entry points.
- No API keys committed. If future stages need secrets, use `.env` + `python-dotenv`, gitignored.

---

## 8. How to Work Each Session

At the start of every session:
1. Run `git status` and `git log --oneline -20`.
2. Read the most recent `docs/plans/stageN-status.md`.
3. Restate in one short paragraph: where we are, what stage is active, what the next concrete task is.
4. If the next task is ambiguous, propose a plan and wait for confirmation before writing code.

At the end of every session:
1. Run `make lint && make typecheck && make test`.
2. If anything fails that was passing before this session, fix it or explicitly document the regression in the status doc.
3. Update `docs/plans/stageN-status.md` with what was done, what's left, known issues.
4. Commit with a clear message. Do not leave a dirty working tree unless explicitly told to.

---

## 9. When You're Stuck

If a dataset is unreachable, a model weight is gone, a library version conflicts, or a task is under-specified:
1. Stop coding.
2. Write the specific blocker to the current stage's status doc.
3. Propose 2–3 options with tradeoffs.
4. Ask the user which to pursue.

Do not paper over blockers with fake data, commented-out code, or "TODO: fix later" that silently breaks the pipeline.

---

## 10. First Action

Your first action in the empty directory is:
1. Print a one-paragraph understanding of this document.
2. Run `git init` and create `main`.
3. Begin Stage 0 on branch `stage-0-bootstrap`.
4. Stop after Stage 0's exit criterion is met and report status before touching Stage 1.

Good luck. Build it carefully.
