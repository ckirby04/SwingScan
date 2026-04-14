# Stage 3 — GolfDB + Pro Reference Bank: status

**Branch:** `stage-3-probank`. **Base:** Stage 2.

## Exit criterion (§5.3)

> `python scripts/build_pro_bank.py --limit 20` runs end-to-end on
> whatever GolfDB data is available, producing a queryable bank file.
> Count of successfully processed swings is logged and written to
> `data/pro_bank/manifest.json`.

**Met structurally.** The integration test
`tests/integration/test_build_pro_bank.py` runs `build_pro_bank` against
the synthetic Stage 1 fixture with a hand-crafted `SwingLabel`, writes a
`bank.parquet` + `manifest.json`, and reloads them through
`ProBank.load`. The script also runs cleanly on an empty labels file,
producing a zero-swing bank and a manifest that records the zero count.

**Real GolfDB data is not downloaded in this session.** The user must
run `scripts/download_golfdb.py` themselves to fetch the annotation
pickle. The video clips are explicitly out of scope for automatic
download per `CLAUDE.md` §6.

## What was built

- `src/swingscan/phases/events.py` — `SwingEvent` enum (8 canonical
  events in canonical temporal order; also an `index` property).
- `src/swingscan/compare/pro_bank.py` — `SwingLabel` (pydantic),
  `ProSwing` (frozen dataclass), `ProBank` with `load` / `save` /
  `filter` / `event_poses` / `from_rows`. `build_pro_bank(labels,
  pose_backend_factory=None)` iterates labels, runs pose, extracts per-
  event snapshots, skips bad rows with clear warnings.
- `scripts/download_golfdb.py` — fetches the GolfDB annotation pickle
  from the upstream GitHub repo (no videos). Supports `--dry-run` and
  `--output`.
- `scripts/build_pro_bank.py` — CLI that reads a labels JSON, runs
  `build_pro_bank`, writes bank + manifest, supports `--limit` for
  smoke testing.
- Tests: `tests/unit/test_pro_bank.py` (filter, query, parquet round-
  trip, missing-file error), `tests/integration/test_build_pro_bank.py`
  (end-to-end against synthetic fixture + empty-labels path).

## Totals

71 tests pass, 87.3 % branch coverage. `compare/pro_bank.py` at 91 %.

## Known carry-overs

- **No real GolfDB data on disk.** Users must run the download script.
- **`build_pro_bank` produces all-zero pose snapshots** for the synthetic
  fixture since the fixture has no human. Structurally sound; real data
  will produce real numbers.
- **Pro bank does not (yet) compute joint-angle time series.** Per ADR
  plan, the bank stores per-event pose snapshots and Stage 5 computes
  angles on demand. This is a minor deviation from the `CLAUDE.md` §5.3
  wording but preserves forward compatibility.

## Next

Stage 4 — phase segmentation. Heuristic backend (wrist-velocity zero-
crossings) first; SwingNet wrapper with graceful missing-weights error
second. Stage 0 smoke test behavior of `swingscan phases` subcommand is
unchanged (not wired yet).
