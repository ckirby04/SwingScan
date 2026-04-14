# ADR 001 — Split heavy pipeline dependencies into an optional extra

**Status:** Accepted
**Date:** 2026-04-14
**Stage:** 0

## Context

`CLAUDE.md` §3 pins the full core library stack: `numpy`, `scipy`,
`opencv-python`, `mediapipe`, `torch`, `torchvision`, `ultralytics`, `pandas`,
`scikit-learn`, plus `gradio` later for the demo UI. Together these pull
roughly 3–4 GB of wheels on a fresh install, dominated by `torch` and
`mediapipe`.

Stage 0's exit criterion is `make install && make lint && make typecheck &&
make test` on a clean checkout. None of those targets actually import the
heavy ML libraries — the smoke test only imports the `swingscan` package and
checks `__version__`. Pinning the full stack as main-project dependencies
would mean:

- every CI run (or fresh clone) downloads multiple gigabytes,
- a wheel resolution failure in any single heavy package blocks Stage 0
  verification,
- contributors cannot iterate on lint/type/test for pure-Python modules
  without waiting on the full stack.

## Decision

Split dependencies into three groups in `pyproject.toml`:

1. **Core (`[project].dependencies`)** — lightweight, pure-data libraries the
   package needs at import time even in early stages: `numpy`, `pandas`,
   `pydantic`, `pyyaml`, `pyarrow`.
2. **Pipeline extra (`[project.optional-dependencies].pipeline`)** — the
   heavy vision/ML stack: `opencv-python`, `scipy`, `scikit-learn`,
   `mediapipe`, `torch`, `torchvision`, `ultralytics`, `gradio`. Installed
   via `pip install -e ".[pipeline]"` starting in Stage 1.
3. **Dev extra (`[project.optional-dependencies].dev`)** — `pytest`,
   `pytest-cov`, `ruff`, `mypy`, `pre-commit`, `types-PyYAML`.

`make install` runs `pip install -e ".[dev]"` for the Stage 0 bootstrap, and
will be updated in Stage 1 to include `[pipeline]` as well.

All versions remain pinned exactly as listed, so the pin strength promised
in `CLAUDE.md` §3 is preserved — only the install boundary changes.

## Consequences

**Positive**
- Stage 0 `make install` completes in seconds instead of minutes.
- Lint/type/test can run in constrained environments (no CUDA, no large
  disk) during early stages.
- Failures in heavy wheels (a common source of Windows install pain) do not
  block green-field verification of Stage 0.

**Negative**
- A contributor who runs `pip install swingscan` (without an extra) gets a
  package that cannot execute the actual pipeline. The `README` and the
  `Makefile` must document which extras are needed for which stages.
- The `Makefile`'s `install` target changes semantics at the Stage 0 → 1
  boundary. This is a one-time edit, not an ongoing hazard.

## Follow-ups

- Stage 1 plan: update `Makefile` `install` target to
  `pip install -e ".[dev,pipeline]"` and add a note in
  `docs/plans/stage1-plan.md`.
- When CI arrives, cache the `[pipeline]` wheels aggressively.
