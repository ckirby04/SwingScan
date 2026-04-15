"""Annotated-video overlay rendering.

:mod:`swingscan.viz.overlay` draws the pose skeleton, phase banner,
and optional club-head trail on each frame of the source video, then
writes an mp4 via OpenCV's ``VideoWriter`` with a preferred-codec
chain (mp4v → avc1 → MJPG). The club trail is off by default; pass
``swingscan run --draw-club`` to include it.
"""
