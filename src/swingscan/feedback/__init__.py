"""Feedback rule engine and human-readable renderer.

:mod:`swingscan.feedback.rules` loads ``configs/feedback_rules.yaml``,
evaluates each rule against a ``SwingDiff``, and emits sorted
``FeedbackItem`` objects. :mod:`swingscan.feedback.render` turns that
list into a plain-text coaching report or a JSON payload. The shipped
rules speak in plain coach voice — see the YAML for the
problem/feel/drill template used by every rule.
"""
