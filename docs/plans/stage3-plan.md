# Stage 3 — GolfDB Ingestion + Pro Reference Bank: plan

**Branch:** `stage-3-probank` off `main` @ Stage 2 (`3678259`).
**Spec:** `CLAUDE.md` §5.3.

## Scope

Implement the code path so that a Stage 5 comparison has something to
query, while respecting that we cannot bundle GolfDB videos or YouTube
content. The tests use a **fabricated mini-bank** built from the
synthetic Stage 1 fixture, since it is a permissibly-licensed stand-in
for the real thing (and all detections are zero-confidence, which is
fine — the bank cares about structure, not golf quality).

The real GolfDB pipeline will work from the same code when the user
opts in to the download script.

## Files

### Source

| Path | Purpose |
|---|---|
| `src/swingscan/compare/pro_bank.py` | `SwingLabel` pydantic model (view, handedness, club, event frame indices, video path, swing_id). `ProSwing` dataclass — one swing's per-event pose snapshots. `ProBank` — load/filter/query. Parquet primary; JSON sidecar for `manifest.json`. |

### Scripts

| Path | Purpose |
|---|---|
| `scripts/download_golfdb.py` | Fetches the GolfDB annotation pickle from the upstream GitHub repo. `--dry-run` prints what would be downloaded without touching the network. Does not auto-fetch videos. |
| `scripts/build_pro_bank.py` | Takes a labels file + video directory, runs the pose pipeline on each swing, emits per-event `ProSwing` rows, writes the bank parquet and a `manifest.json` summary. |

### Tests

| Path | What |
|---|---|
| `tests/unit/test_pro_bank.py` | Fabricate three synthetic `ProSwing` rows; write/read bank round-trip; filter by view/handedness/club; query by event name. |
| `tests/integration/test_build_pro_bank.py` | Build a mini-bank from `sample_swing.mp4` with a hand-crafted `SwingLabel`, assert manifest + bank file are produced and reloadable. |

## Exit criterion

`python scripts/build_pro_bank.py --limit N` runs end-to-end on whatever
labels are provided, producing a queryable `bank.parquet` and a
`manifest.json`. Count of successfully processed swings is logged and
written to the manifest.

When no real data is available, the script succeeds with zero swings and
writes an empty manifest, logging a clear note pointing at
`scripts/download_golfdb.py`.
