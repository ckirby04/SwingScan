"""Pose estimation.

:mod:`swingscan.pose.keypoints` defines the 33-joint ``Joint`` enum and
semantic groupings. :mod:`swingscan.pose.base` holds the ``PoseFrame``
/ ``PoseSequence`` dataclasses and the ``PoseEstimator`` protocol.
:mod:`swingscan.pose.mediapipe_backend` is the default concrete
backend, wrapping MediaPipe BlazePose.
"""
