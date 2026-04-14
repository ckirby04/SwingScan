# Stage 9 — evaluation harness: status

## Exit criterion (§5.9)

> A single command produces a reproducible evaluation report with
> numbers, not placeholders.

**Met structurally.** `python scripts/evaluate_pipeline.py
--labels X --output report.json` runs the pose → heuristic-segmenter
path over each labeled swing, computes PCE (with configurable tolerance),
pose completeness, and — when `--bank` is supplied — pro-bank coverage,
and writes a JSON report.

When the labels file is missing, the script writes a `no_data`
report rather than crashing (integration-tested). When the labels point
at a real video the script produces real PCE / completeness numbers —
demonstrated on the synthetic fixture.

## What was built

- `scripts/evaluate_pipeline.py` — argparse CLI + `_score_swing()`
  per-swing helper + JSON report writer. Handles missing videos,
  empty labels, and missing bank files gracefully.
- Tests: `tests/integration/test_evaluate_pipeline.py` (empty labels
  → no_data report; synthetic fixture → scored report with numeric
  PCE).

104 tests pass, 84.9 % branch coverage.

## Known carry-overs

- **No real numbers against GolfDB.** Users must run the download
  script themselves, wire up a labels JSON pointing at the clips, and
  run `python scripts/evaluate_pipeline.py --labels ...`. The reported
  PCE against the heuristic segmenter is expected to be modest (the
  segmenter is a wrist-velocity heuristic, not SwingNet); this is the
  intended baseline.
- **No aggregation across runs.** Each report is a one-off snapshot.
  Stage 10 polish can add a rolling history if useful.

## Next

Stage 10 — polish, `ROADMAP.md`, `v0.1.0` tag, README/architecture
writing.
