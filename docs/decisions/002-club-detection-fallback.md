# ADR 002 — Club-head detection fallback strategy

**Status:** Accepted
**Date:** 2026-04-14
**Stage:** 2

## Context

`CLAUDE.md` §5.2 mandates two club-head backends:

1. A **YOLOv8 nano** model fine-tuned on labeled frames
   (`models/club_yolo.pt`), loaded on demand.
2. A **classical / heuristic fallback** when the weights file is absent,
   so the pipeline has no hard download dependency.

It also explicitly asks for an ADR on:

- How the fallback works.
- What "good enough" means for club tracking in V1.

## Decision

### Fallback backend: pose-derived geometric proxy

The V1 fallback is `HeuristicClubDetector`. It does **not** look at
pixels. Instead, for every frame it reads the pose estimate and computes:

1. Wrist midpoint `M` in image-normalized space.
2. Forearm direction — the unit vector from the shoulder midpoint to
   `M`.
3. Club-head position: `M` plus `club_length_shoulder_ratio * shoulder_width`
   along that direction. Default ratio is `2.0`, which roughly matches a
   driver club length for a face-on address posture at typical consumer
   video aspect ratios.
4. Confidence: the product of the two wrist visibilities.

Rejected alternatives:

- **Motion + color + line detection near the hands.** The section in
  `CLAUDE.md` §5.2 lists this as an option. We rejected it because:
  - It requires per-video color/line tuning (white shafts vs black vs
    chrome all behave differently under arbitrary lighting).
  - It has a high false-positive rate against the pants, ground shadow,
    and club bag.
  - The motion signal is unreliable during address and finish poses
    (by definition — the club is motionless).
  - Tuning is not testable without a labeled dataset, which we explicitly
    do not require in Stage 2.
- **Optical-flow tracking seeded from the wrist midpoint.** More robust
  than the geometric proxy once moving but still brittle at address and
  adds an opencv-contrib dependency. Reserved for V2.

### "Good enough" definition for V1

V1 is explicitly not a shot-shape or launch-monitor product
(`CLAUDE.md` §2). The club trajectory exists to:

1. Disambiguate the **top** of the backswing and the **impact frame**
   for the Stage 4 phase segmenter when its heuristic mode is active.
2. Support the viz overlays in Stage 7 (draw a swept arc).
3. Feed Stage 5 metrics that care about the hand path more than the club
   itself (e.g., swing plane proxy).

With those goals, "good enough" means:

- **Coverage:** every frame gets a `ClubDetection`. Missing detections
  are tagged `source="missing"` rather than dropped.
- **Monotonicity of trajectory during the swing:** when the golfer is
  visible and wrists are above the minimum visibility threshold, the
  smoothed club position moves in a direction roughly consistent with
  the hand path (unit tests enforce this structurally on synthetic
  poses).
- **Graceful failure:** if the pose backend returns low-confidence
  wrists (as with the Stage 1 synthetic fixture, which depicts no real
  human), the pipeline does not crash. Missing runs longer than
  `tracker.max_gap_frames` stay missing, with a warning logged.

We explicitly do **not** require:

- A quantitative positional error bound against real club trajectories.
  That requires a labeled dataset we don't have yet.
- Real-time performance.
- Handedness awareness at this stage.

### Temporal smoothing: exponential, not Kalman

`CLAUDE.md` §5.2 suggests "Kalman filter or simple exponential smoothing."
We pick exponential smoothing with `alpha=0.4` plus linear gap-fill for
runs ≤ 3 frames. The Kalman formulation adds state-space machinery that
we don't need given the low frame rate and short clips — the club head
is never occluded for more than a few frames when the golfer is visible,
and when it is (for more than 3), a Kalman filter would confidently drift
in the wrong direction.

## Consequences

- **Positive:** zero external downloads, heuristic runs on any CPU, code
  is tiny and readable, unit-testable on fabricated poses.
- **Negative:** positions are a geometric proxy, not a true detection.
  Stage 5 metrics that care about absolute club position (e.g., club
  face angle) cannot rely on the heuristic. They must either short-circuit
  to "unavailable" or require the YOLO backend.
- **Follow-up:** revisit in V2. Train a real YOLOv8 nano club detector on
  a labeled subset of GolfDB after the Stage 3 dataset scripts land.
