# SwingScan Pipeline Reference

> **Status:** placeholder. This document is filled in during Stage 1 (for
> the pose schema) and extended as each later stage lands.

Eventual contents:

- Exact schemas for on-disk artifacts (`PoseSequence` parquet layout,
  phase JSON, `SwingDiff` structure, feedback report JSON).
- The step-by-step flow through `swingscan.pipeline` for a single input
  video.
- How optional components (GolfDB pro bank, SwingNet weights, YOLO club
  detector) degrade gracefully when absent.

Until Stage 1 ships, refer to `CLAUDE.md` §5.1 for the intended Stage 1
surface area.
