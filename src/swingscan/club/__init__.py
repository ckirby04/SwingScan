"""Club-head detection and tracking.

:mod:`swingscan.club.detector` provides two backends: a pose-derived
geometric ``HeuristicClubDetector`` (default, no weights required)
and a ``YoloClubDetector`` that activates when a trained
``models/club_yolo.pt`` file is present.
:mod:`swingscan.club.tracker` applies causal exponential smoothing
and short-gap linear interpolation to the raw per-frame detections.
"""
