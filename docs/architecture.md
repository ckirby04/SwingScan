# SwingScan Architecture

> **Status:** placeholder. This document is filled in progressively as each
> stage lands and then finalized in Stage 10 per `CLAUDE.md` §5.

The canonical architecture reference for now is `CLAUDE.md` §2–§4 (vision,
scope, tech stack, and repository layout). This file will eventually hold:

- A high-level data-flow diagram from raw video → keypoints → phases →
  metrics → feedback.
- Module responsibilities and the boundaries between them.
- Performance budgets per stage (e.g., pose @ N fps on CPU, N fps on GPU).
- Extension points for V2 work (learned swing embeddings, 3D lifting,
  multi-view fusion).

Until Stage 10, defer to `CLAUDE.md` and the `docs/plans/stageN-*.md` files.
