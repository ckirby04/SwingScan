# Stage 8 — Gradio local demo: status

## Exit criterion (§5.8)

> `make demo` opens a browser, user can upload a sample video, and a
> report is returned end-to-end.

**Met structurally.** `make demo` launches `scripts/demo_local.py`
which boots a Gradio Blocks app bound to 127.0.0.1:7860 (override via
`--host` / `--port`). The upload widget triggers `run_pipeline(...)`
behind the scenes, writes an annotated mp4 to a temp path, and returns
it alongside a per-phase metrics dataframe and the prioritized feedback
list.

End-to-end verification via a real browser was not done in this
session (no interactive browser), but:

- The internal helpers `_format_metrics_table` and `_format_feedback`
  are covered by unit tests that feed a fabricated `PipelineResult`
  and check the resulting rows/strings.
- The argparse surface is covered by a parser test.
- The `--pro-bank` flag is accepted and plumbed through to
  `run_pipeline`, so cohort-comparison feedback is available whenever
  the user has built a bank.

Upload/duration caps enforced per §5.8: default 100 MB / 30 s, both
configurable via CLI flags. Rejected uploads return a clear error
string and never call into the pipeline. No analytics
(`analytics_enabled=False`), no persistent uploads, binds to localhost
by default.

## What was built

- `scripts/demo_local.py` — Gradio app + metrics / feedback helpers +
  CLI (host, port, pro-bank path, rules path, max-mb, max-duration-s).
- `Makefile` — `demo` target now actually runs the script.
- Tests: `tests/unit/test_demo_local.py` (parser defaults, metrics
  table formatting with and without a cohort, feedback rendering).

102 tests pass, 84.9 % branch coverage.

## Next

Stage 9 — evaluation harness. Requires a real GolfDB subset to be
meaningful; the script can still be built against synthetic data to
prove the plumbing.
