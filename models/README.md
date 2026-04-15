# models/

This directory is gitignored. It holds pretrained weights and fine-tuned
checkpoints used by the pipeline at runtime.

Expected contents (populated as needed, not committed):

- `swingnet_1800.pth.tar` — SwingNet weights from the upstream GolfDB
  release. Distributed by the GolfDB authors under CC BY-NC 4.0.
  Download via `gdown`:

  ```bash
  python -m gdown \
    "https://drive.google.com/uc?id=1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW" \
    -O models/swingnet_1800.pth.tar
  ```

  `SwingNetSegmenter` auto-discovers this path. Without it, the
  pipeline falls back to the wrist-velocity `HeuristicSegmenter`
  (PCE ~9.5 % on our eval, versus ~94.75 % with SwingNet).

- `club_yolo.pt` — placeholder for a fine-tuned YOLOv8n club-head
  detector. Not shipped by anyone today; training one is V2 work.
  When present, `YoloClubDetector` uses it; when absent, the pipeline
  falls back to the pose-derived `HeuristicClubDetector`.

Do not commit any weight file to git — see `.gitignore`.
