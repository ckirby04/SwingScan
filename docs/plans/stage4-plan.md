# Stage 4 — Phase Segmentation: plan

**Branch:** `stage-4-phases` off main @ Stage 3. **Spec:** `CLAUDE.md` §5.4.

## Goal

Given a `PoseSequence`, map each of the 8 `SwingEvent` values to a frame
index, monotonically in temporal order. Two backends:

1. **`HeuristicSegmenter`** — pose-based, wrist velocity + hand position
   zero-crossings. Runs with no external weights. Primary backend for V1.
2. **`SwingNetSegmenter`** — loads SwingNet weights when present; otherwise
   raises `FileNotFoundError` pointing at `models/README.md`.

## Files

- `src/swingscan/phases/segmenter.py` — `PhaseSegmenter` protocol,
  `HeuristicSegmenter`, `PhaseMap` dataclass.
- `src/swingscan/phases/swingnet.py` — `load_swingnet` + `SwingNetSegmenter`
  (weight-gated).
- `src/swingscan/cli.py` — new `phases` subcommand that runs pose →
  segment and writes a JSON phase map.
- `tests/unit/test_heuristic_segmenter.py` — drive synthetic pose sequences
  through the segmenter and assert event ordering + expected-frame
  tolerance.
- `tests/unit/test_swingnet_stub.py` — assert missing-weights error.

## Heuristic algorithm

1. Extract wrist-midpoint y position across all frames: `y(t)` in
   image-normalized space.
2. Compute wrist speed: `v(t) = |pos(t) - pos(t-1)|`.
3. **Address:** first frame where both wrists have visibility ≥
   threshold. Fallback: frame 0.
4. **Top of backswing:** argmin of wrist-midpoint y (highest hands,
   smallest y in image coords).
5. **Impact:** argmax of wrist speed between `top` and the end.
6. **Toe_up:** frame roughly halfway between address and top along the
   wrist y trajectory.
7. **Mid_backswing:** frame halfway between toe_up and top.
8. **Mid_downswing:** frame halfway between top and impact.
9. **Mid_follow_through:** frame halfway between impact and end.
10. **Finish:** last non-low-confidence frame, or the last frame period.

Monotonicity is enforced after the fact; if any event index is out of
order, the segmenter clamps the offending value to `prev + 1`.

## Synthetic-fixture tolerance

The Stage 1 synthetic fixture has no real human, so the heuristic's
inputs are zero-confidence — the segmenter falls back to evenly-spaced
events across the clip length. The integration test asserts only
"8 events in monotonic order, covering frame 0 through last frame," not
physically meaningful correctness. Real videos get a more interesting
test later.

## Exit criterion

Both backends run. The heuristic runs with no external weights. Running
`swingscan phases` on the fixture produces all 8 events in monotonic
order.
