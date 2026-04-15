# data/

This directory is gitignored. It holds inputs and derived artifacts that are
too large, too licensed, or too ephemeral to live in version control.

Subdirectories (created by Stage 1+ scripts):

- `raw/` — unmodified inputs: downloaded GolfDB annotations, user-supplied
  swing videos, optionally (if the user opts in via `yt-dlp`) the source
  YouTube videos referenced by GolfDB. Never committed.
- `processed/` — per-video derived artifacts (pose parquet, phase JSON,
  annotated videos) produced by the pipeline.
- `pro_bank/` — the compiled professional reference bank produced by
  `scripts/build_pro_bank.py`. Quality is bounded by input resolution; if you
  build from the 160x160 GolfDB clips, expect noisy pose data. See Stage 3 in
  `CLAUDE.md` for the quality/licensing tradeoffs.

**Licensing reminder:** Do not redistribute GolfDB videos or YouTube-sourced
content. Everything downstream of those sources stays on-device.
