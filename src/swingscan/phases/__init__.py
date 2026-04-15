"""Swing-phase segmentation.

:mod:`swingscan.phases.events` defines the 8-event ``SwingEvent``
enum. :mod:`swingscan.phases.segmenter` provides the
``PhaseSegmenter`` protocol and the wrist-velocity
``HeuristicSegmenter``. :mod:`swingscan.phases.swingnet` wraps the
vendored upstream SwingNet model (``_swingnet_model.py``) and
becomes the default segmenter when
``models/swingnet_1800.pth.tar`` is present — see
:mod:`swingscan.phases._swingnet_model` for attribution.
"""
