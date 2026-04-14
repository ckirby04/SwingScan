"""Tests for the RuleEngine and feedback renderer."""

from __future__ import annotations

from pathlib import Path

import pytest

from swingscan.compare.diff import MetricDelta, PhaseDiff, SwingDiff
from swingscan.feedback.render import render_json, render_text
from swingscan.feedback.rules import FeedbackItem, RuleEngine


def _diff_with(event: str, metric: str, delta: float, amateur: float = 60.0) -> SwingDiff:
    metric_delta = MetricDelta(
        metric=metric,
        amateur=amateur,
        cohort_median=amateur - delta,
        delta=delta,
        z_score=delta / 5.0,
    )
    phase = PhaseDiff(event=event, metrics=(metric_delta,))
    return SwingDiff(phases=(phase,), cohort_size=10)


def _rules_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "rules.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_rule_fires_when_delta_exceeds_threshold(tmp_path: Path) -> None:
    rules = _rules_yaml(
        tmp_path,
        """
rules:
  - name: test_above
    phase: TOP
    metric: shoulder_rotation_deg
    direction: above
    threshold: 10
    severity: 2
    message_template: "shoulder is {delta:.0f} past cohort"
""",
    )
    engine = RuleEngine.from_yaml(rules)
    diff = _diff_with("TOP", "shoulder_rotation_deg", delta=15)
    items = engine.evaluate(diff)

    assert len(items) == 1
    assert items[0].rule == "test_above"
    assert "15" in items[0].message


def test_rule_below_threshold_is_silent(tmp_path: Path) -> None:
    rules = _rules_yaml(
        tmp_path,
        """
rules:
  - name: test_above
    phase: TOP
    metric: shoulder_rotation_deg
    direction: above
    threshold: 10
    severity: 2
    message_template: "no fire"
""",
    )
    engine = RuleEngine.from_yaml(rules)
    diff = _diff_with("TOP", "shoulder_rotation_deg", delta=5)
    assert engine.evaluate(diff) == []


def test_rule_below_direction_fires_on_negative(tmp_path: Path) -> None:
    rules = _rules_yaml(
        tmp_path,
        """
rules:
  - name: test_below
    phase: IMPACT
    metric: hip_rotation_deg
    direction: below
    threshold: -10
    severity: 3
    message_template: "hips short by {delta:.0f}"
""",
    )
    engine = RuleEngine.from_yaml(rules)
    diff = _diff_with("IMPACT", "hip_rotation_deg", delta=-12, amateur=5)
    items = engine.evaluate(diff)
    assert len(items) == 1
    assert items[0].severity == 3


def test_rules_yaml_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RuleEngine.from_yaml(tmp_path / "nope.yaml")


def test_default_rules_yaml_loads_and_parses() -> None:
    engine = RuleEngine.from_yaml()
    # configs/feedback_rules.yaml should have at least 10 rules.
    assert len(engine.rules) >= 10


def test_render_text_empty_returns_placeholder() -> None:
    out = render_text([])
    assert "No feedback" in out


def test_render_text_caps_and_warns() -> None:
    items = [
        FeedbackItem(
            rule=f"r{i}",
            phase="TOP",
            metric="m",
            severity=3,
            message=f"cue {i}",
            amateur=0.0,
            delta=1.0,
            z_score=1.0,
        )
        for i in range(8)
    ]
    out = render_text(items, max_items=3)
    assert "cue 0" in out
    assert "cue 2" in out
    assert "cue 3" not in out
    assert "+5 additional cues" in out


def test_render_json_has_disclaimer() -> None:
    import json

    out = render_json([])
    payload = json.loads(out)
    assert payload["format"] == "swingscan_feedback_v1"
    assert "disclaimer" in payload
