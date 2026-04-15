# SwingScan Roadmap

This file lists follow-on work beyond the V1 shipped at `v0.1.0`. See
[`CLAUDE.md`](./CLAUDE.md) §5 for what V1 covers.

## Landed post-v0.1.0

- **Real SwingNet inference.** `HeuristicSegmenter` replaced as the
  default when `models/swingnet_1800.pth.tar` is present. PCE on our
  50-swing face-on driver eval jumped from **9.5 %** to **94.75 %**
  (tol=5) / **82.5 %** (tol=1). Heuristic remains as the no-weights
  fallback. See `src/swingscan/phases/_swingnet_model.py` (vendored,
  CC BY-NC 4.0 attribution preserved) and
  `docs/evaluations/eval_50_face_on_driver_swingnet.json`.
- **Circular statistics for angular metrics.** `compare/diff.py` now
  uses circular-mean central tendency and wraps deltas into
  [-180, 180]. Fixes the "shoulder rotation 127° past cohort" false
  alarm that stemmed from naive arithmetic-mean at the ±180° boundary.
- **Feedback thresholds calibrated from the real bank.**
  `scripts/calibrate_thresholds.py` rewrites
  `configs/feedback_rules.yaml` with per-metric 1.5σ thresholds
  measured against the 178-swing face-on driver bank. Jerry Kelly's
  feedback dropped from 5 cues (4 false positives) to 1 clean cue.
- **UTF-8 stdout in the CLI.** `_force_utf8_stdio()` fixes Windows
  PowerShell rendering of °, —, and ·.
- **GitHub Actions CI.** `.github/workflows/ci.yml` runs lint, ruff
  format check, strict mypy, and pytest on every push + PR against
  `main`.
- **Down-the-line pro bank.** A second cohort under
  `data/pro_bank_dtl/` built from 225 GolfDB down-the-line driver
  swings, for comparing against DTL uploads.

## V1.1 — still open

- Train a small YOLOv8 nano club-head detector on a hand-labeled
  subset of GolfDB frames and replace `HeuristicClubDetector` with it
  as the primary. Ship weights in a release asset, not git.
- Publish a real-human fixture video under a permissive license so the
  integration tests can assert non-degenerate metrics and confidence
  ratios.
- Filter the SwingNet eval to GolfDB split 4 only so the reported PCE
  is an honest held-out number (current 82.5 % at tol=1 is inflated
  by training-set overlap with the upstream pretrain).
- Handedness detection — today the pipeline defaults to `right`. A
  trivial check (which wrist is inside of the other at address) would
  cover 99 % of cases.
- View auto-detect — face-on vs down-the-line from the first few
  frames so the caller doesn't have to pass `--pro-bank` explicitly.
- Persistent browser test for `make demo`, ideally via Playwright in
  CI against a committed test clip.

## V2 — richer signal

- **Learned swing embeddings.** Train a small contrastive model on
  GolfDB pose trajectories, use nearest-neighbor retrieval against the
  pro bank instead of hand-picked per-metric z-scores. Rule engine
  remains for explainability but is augmented by model-driven cues.
- **3D lifting.** Replace the 2D pose proxy with a 3D lifter
  (VideoPose3D, MotionBERT, HybrIK) so biomech metrics become actually
  physical. Especially useful for spine tilt, x-factor, and weight
  shift.
- **Multi-view fusion.** Support combining face-on and down-the-line
  simultaneously when both are available; drastically improves club
  plane and wrist hinge estimates.
- **Mobile deployment.** Package the V2 pipeline behind an ONNX /
  Core ML bundle for on-device inference. Not a web service — on-
  device keeps the licensing and privacy story simple.

## Infrastructure

- ~~Continuous integration (GitHub Actions) running lint, typecheck,
  test on push + PR.~~ **DONE**, post-v0.1.0.
- Release automation — tag → PyPI wheel + annotated GitHub release.
- Cache the `[pipeline]` extras in CI. ~4 GB of wheels; cold installs
  are expensive. Current CI does cache pip via
  `cache-dependency-path: pyproject.toml` but wheels still rebuild on
  pin bumps; a wheel-cache step would cut CI cold-start time further.

## Known tech debt

- `VideoReader` re-iterates the source video twice when the YOLO club
  backend is selected. Stage 7 overlays also walk the video a second
  time. SwingNet adds a third video walk. A unified "walk the video
  once, fan out to all consumers" primitive would cut CPU time roughly
  by 3x on the SwingNet path.
- `configs/pose_mediapipe.yaml` and `configs/phases_swingnet.yaml` are
  empty placeholders — the schemas live in code for now. Populate them
  when the underlying backends get real knobs worth exposing (SwingNet
  seq_length, MediaPipe model_complexity, etc.).
- SwingNet inference holds all preprocessed frames in a single tensor
  before batching. For long clips this OOMs; acceptable for GolfDB's
  ~100-frame clips but needs streaming preprocessing before V2 mobile
  deployment.
