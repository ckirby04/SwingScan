# Stage 5 — metrics + SwingDiff: status

## Exit criterion (§5.5)

> Given a pose sequence + phase map + pro bank, the pipeline produces a
> `SwingDiff` with at least 8 computed metrics per phase and z-scores.

**Met.** `compare_against_bank(pose, phases, bank)` returns a
`SwingDiff` where each `PhaseDiff` has 8 `MetricDelta` entries
(`hip_rotation`, `shoulder_rotation`, `x_factor`, `spine_lean`,
`lead_arm_angle`, `wrist_hinge`, `lead_knee_flex`, `head_movement`).
Z-scores are computed against the bank cohort when there are ≥ 2 swings;
otherwise they are `None`.

## What was built

- `src/swingscan/metrics/angles.py` — `signed_angle_deg`,
  `hip_rotation_deg`, `shoulder_rotation_deg`, `x_factor_deg`,
  `spine_lean_deg`, `lead_arm_straightness_deg`, `wrist_hinge_deg`,
  `knee_flex_deg`, `head_movement_px`, `shoulder_width`, `torso_length`.
- `src/swingscan/metrics/normalize.py` — `normalize_frame` (scale by
  shoulder-width or torso-length, center on hip midpoint),
  `flip_handedness` (mirror + swap LEFT/RIGHT pairs).
- `src/swingscan/metrics/biomech.py` — `PhaseMetrics`, `SwingMetrics`,
  `compute_swing_metrics(pose, phase_map, handedness)`.
- `src/swingscan/compare/diff.py` — `MetricDelta`, `PhaseDiff`,
  `SwingDiff`, `compare_against_bank(amateur_pose, phases, bank)`.
  Internally reconstructs `SwingMetrics` for each pro in the bank from
  the bank's per-event snapshots, then produces per-metric medians and
  z-scores.
- Tests: `test_angles.py` (8 unit tests with fabricated joint layouts),
  `test_biomech_and_diff.py` (synthetic swing produces 8 phase metrics,
  empty bank yields null deltas, populated bank with the same swing
  yields zero z-scores).

89 tests pass, 84.4 % branch coverage.

## Next

Stage 6 — feedback rule engine + YAML rules + renderer.
