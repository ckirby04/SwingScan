# SwingScan Roadmap

This file lists follow-on work beyond the V1 shipped at `v0.1.0`. See
[`CLAUDE.md`](./CLAUDE.md) §5 for what V1 covers.

## V1.1 — tighten the heuristics

- Replace `HeuristicSegmenter` with a proper SwingNet inference path
  (wrap the GolfDB weights, run on a 160×160 resized clip, post-process
  the logits into the 8 events). Keep the heuristic as the no-weights
  fallback.
- Train a small YOLOv8 nano club-head detector on a hand-labeled
  subset of GolfDB frames and replace `HeuristicClubDetector` with it
  as the primary. Ship weights in a release asset, not git.
- Publish a real-human fixture video under a permissive license so the
  integration tests can assert non-degenerate metrics and confidence
  ratios.
- Run Stage 9 evaluation against 20 hand-labeled GolfDB swings and
  commit the resulting report under `docs/evaluations/`.

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

- Continuous integration (GitHub Actions) running `make lint`,
  `make typecheck`, `make test` on push + PR. Not blocking the `v0.1.0`
  tag; worth doing before onboarding a second contributor.
- Release automation — tag → PyPI wheel + annotated GitHub release.
- Cache the `[pipeline]` extras in CI. ~4 GB of wheels; cold installs
  are expensive.

## Known tech debt

- `VideoReader` re-iterates the source video twice when the YOLO club
  backend is selected. Stage 7 overlays also walk the video a second
  time. A unified "walk the video once, fan out to all consumers"
  primitive would cut CPU time roughly in half.
- `SwingNetSegmenter.segment()` is `NotImplementedError`; the wrapper
  exists only to surface the missing-weights error clearly. Wiring
  real inference is V1.1 work.
- `feedback_rules.yaml` thresholds are hand-tuned against intuition,
  not measured against a distribution. V1.1 should recalibrate them
  once real pro bank metrics exist.
- `configs/pose_mediapipe.yaml` and `configs/phases_swingnet.yaml` are
  empty placeholders — the schemas live in code for now. Populate them
  when the underlying backends get real knobs worth exposing.
