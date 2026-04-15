"""Biomechanical metrics and pose normalization.

:mod:`swingscan.metrics.angles` holds per-frame joint-angle
primitives (hip/shoulder rotation, x-factor, spine lean, lead arm
straightness, wrist hinge, knee flex, head drift).
:mod:`swingscan.metrics.normalize` scales and recenters pose frames
for comparison. :mod:`swingscan.metrics.biomech` composes the
primitives into ``SwingMetrics`` — per-event + composite values for
a single swing.
"""
