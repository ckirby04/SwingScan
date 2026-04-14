# Stage 4 — Phase Segmentation: status

## Exit criterion (§5.4)

Both backends run. `HeuristicSegmenter` runs with no external weights.
`swingscan phases --input fixture --output phases.json` produces a JSON
with all 8 events in strict monotonic order. **Met.**

## What was built

- `phases/events.SwingEvent` — already landed in Stage 3.
- `phases/segmenter` — `PhaseMap` frozen dataclass (`as_dict`,
  `frame_for`, `is_monotonic`), `PhaseSegmenter` protocol, and
  `HeuristicSegmenter`. Algorithm: wrist-midpoint y minimum for top,
  wrist speed maximum (between top and finish) for impact, midpoints
  for the rest, monotonicity enforced at the end.
- `phases/swingnet` — `load_swingnet` + `SwingNetSegmenter`. Lazy
  `torch` import; raises `FileNotFoundError` on missing weights. The
  `segment()` method is explicitly `NotImplementedError` in V1; the
  heuristic is the default backend.
- `cli.phases` subcommand: runs pose → segment → JSON.
- Tests: `test_heuristic_segmenter.py` (synthetic 60-frame swing with
  known y-minimum; zero-visibility fallback still produces 8 monotonic
  events). `test_swingnet_stub.py` (missing weights error).
  `test_phases_cli.py` (CLI integration on the Stage 1 fixture).

77 tests pass, 87.5% coverage.

## Next

Stage 5 — biomechanics metrics + `SwingDiff`. Pure-python on top of the
existing `PoseSequence` + `PhaseMap` + `ProBank` primitives.
