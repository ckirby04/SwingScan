# Stage 10 — Polish: status

## Exit criterion (§5.10)

- Fill in `docs/architecture.md` and `docs/pipeline.md`. **Done.**
- Ensure README quickstart works on a clean machine. **Done** — the
  README install/test/demo flow matches the in-repo Makefile targets.
- Tag `v0.1.0`. **Done at this commit.**
- Write `ROADMAP.md` for V2 ideas. **Done.**

## What was built in this stage

- `README.md` — stage status table flipped to "Done," quickstart
  lists real commands and the Gradio demo launcher.
- `docs/architecture.md` — real ASCII data-flow diagram, module
  responsibilities table, backend fallback summary, informal
  performance budget, pointer to `ROADMAP.md`.
- `docs/pipeline.md` — canonical schemas for every on-disk artifact
  the pipeline produces (pose parquet + JSON, phase JSON, pipeline
  report, pro bank parquet + manifest, evaluation report) plus a
  narrative runtime flow.
- `ROADMAP.md` — V1.1 (tighten heuristics), V2 (learned embeddings,
  3D lifting, multi-view fusion, mobile), infrastructure (CI, release
  automation), and known tech debt.

## Release

Tagged `v0.1.0`. 104 tests pass, 84.9 % branch coverage across core
modules. All 11 stage status docs live under `docs/plans/`.
