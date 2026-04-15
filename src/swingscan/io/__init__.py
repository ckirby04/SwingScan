"""Video I/O and pose-sequence serialization.

:mod:`swingscan.io.video` wraps OpenCV's ``VideoCapture`` with a typed
:class:`~swingscan.io.video.VideoReader`, and :mod:`swingscan.io.serialize`
round-trips :class:`~swingscan.pose.base.PoseSequence` through parquet
(primary) and JSON (debug).
"""
