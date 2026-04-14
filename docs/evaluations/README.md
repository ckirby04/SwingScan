# Evaluation reports

Committed output of `scripts/evaluate_pipeline.py` against labeled GolfDB
splits. Each report documents the labels, tolerance, bank, and resulting
per-swing scores — version-controlled so we can track regressions.

## Reports

### `eval_50_face_on_driver_heuristic.json`

- **Labels:** first 50 face-on driver non-slow-mo swings from GolfDB
  (produced by `scripts/convert_golfdb_labels.py`).
- **Pro bank:** 178-swing face-on driver bank built from the full
  GolfDB 160×160 release.
- **Segmenter:** `HeuristicSegmenter` (wrist-velocity heuristic, no
  SwingNet weights).
- **Tolerance:** ±5 frames.

**Results:**

| Metric | Value |
|---|---|
| Swings scored | 50 / 50 |
| Pose completeness | 99.6 % |
| Pro-bank coverage | 100 % |
| PCE (±5 frames) | **9.5 %** |

**Analysis.** The heuristic segmenter scores poorly against GolfDB
labels because GolfDB annotates `ADDRESS` as the moment the golfer is
set up and ready to begin — typically 20–40 frames after the start of
the preprocessed clip (which includes pre-address padding). The
heuristic uses "first confident pose frame" for `ADDRESS`, which lands
at frame 0, missing GolfDB's labeled frame by tens of frames on most
clips.

This is a known limitation, documented in `ROADMAP.md` under **V1.1 —
tighten the heuristics** as "Replace `HeuristicSegmenter` with a proper
SwingNet inference path." SwingNet achieves 71.5 % PCE on the same
split (per the GolfDB paper). The downstream pipeline (metrics,
comparison, feedback, viz) is unaffected by segmenter quality in the
structural sense — it just consumes whichever event frames the
segmenter returns. Bad events → noisy feedback; good events → crisp
feedback.

### Reproducing

```bash
python scripts/convert_golfdb_labels.py
.venv/Scripts/python.exe scripts/build_pro_bank.py
.venv/Scripts/python.exe scripts/evaluate_pipeline.py \
    --labels data/raw/labels.json \
    --bank   data/pro_bank/bank.parquet \
    --limit  50 \
    --output docs/evaluations/eval_50_face_on_driver_heuristic.json
```
