# Stage 6 — feedback rules + renderer: status

## Exit criterion (§5.6)

> `swingscan run --input video.mp4` prints a full report: detected
> phases, key metrics, top feedback items.

**Met.** `swingscan run` produces a phases-bullet-plus-feedback report
on stdout. When a `--pro-bank` parquet is supplied, the pipeline runs
the Stage 5 comparison and the Stage 6 rule engine; feedback items are
sorted by severity and rendered with a `!!!`/`!!`/`!` severity marker.
Without a bank, the report still prints — just without populated diff
deltas, and the feedback section reads "No feedback items — pose or
cohort data were insufficient."

## What was built

- `configs/feedback_rules.yaml` — 15 rules covering insufficient/
  excessive turn, hip stall, early extension, lead arm collapse,
  casting, reverse pivot, head dip/sway, knee flex loss, X-factor,
  wrist hinge, hip-early-release, head-up finish. Threshold deltas and
  severities are all in YAML so they're tweakable without code changes.
- `src/swingscan/feedback/rules.py` — `FeedbackRule` pydantic model,
  `FeedbackRuleSet`, `FeedbackItem` dataclass, `RuleEngine.from_yaml()`
  + `evaluate(diff)`. Rules are ordered by severity (high to low) then
  name.
- `src/swingscan/feedback/render.py` — text and JSON renderers with a
  `max_items` cap, disclaimer, and severity markers.
- `src/swingscan/pipeline.py` — extended to run Stage 4 phases
  automatically and, when `--pro-bank` is supplied, Stage 5 diff +
  Stage 6 evaluation. Adds a `render_report()` helper.
- `src/swingscan/cli.py` — `run` subcommand now takes `--pro-bank`,
  `--rules`, `--handedness`. Prints `render_report()` to stdout. When
  `--output` is passed, the JSON payload includes `phases` and
  `feedback` keys.
- Tests: `test_feedback_rules.py` exercises rule-fires-above, below,
  silent, missing-yaml error, the default-rules-yaml load, and text/
  JSON render contracts. Updated `test_pipeline_stage2` to match the
  new "SwingScan report" header.

97 tests pass, 84.7 % branch coverage.

## Next

Stage 7 — viz overlays (OpenCV skeleton draw + club trail + phase
banner). First stage that actually touches the output-video path.
