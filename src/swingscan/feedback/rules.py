"""Rule engine that turns a :class:`SwingDiff` into feedback items.

Rules live in ``configs/feedback_rules.yaml``. Each rule fires when a
specific ``(phase, metric)`` delta crosses a threshold in the specified
direction, producing a :class:`FeedbackItem` at a given severity.

The engine is intentionally symbolic (no ML). Every firing cue is
explainable and traceable to a YAML line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from swingscan.compare.diff import SwingDiff
from swingscan.phases.events import SwingEvent
from swingscan.utils.paths import configs_dir

__all__ = [
    "Direction",
    "FeedbackRule",
    "FeedbackRuleSet",
    "FeedbackItem",
    "RuleEngine",
    "default_rules_path",
]

_log = logging.getLogger(__name__)

Direction = Literal["above", "below"]


class FeedbackRule(BaseModel):
    """Schema for one rule in ``configs/feedback_rules.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    phase: str
    metric: str
    direction: Direction
    threshold: float
    severity: int = Field(ge=1, le=3)
    message_template: str


class FeedbackRuleSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules: list[FeedbackRule]


@dataclass(frozen=True, slots=True)
class FeedbackItem:
    """A single user-visible coaching cue.

    ``message`` is the rendered human string; ``phase``, ``metric``,
    ``severity``, ``amateur``, ``delta``, ``z_score`` are kept for
    downstream renderers (UI cards, JSON payloads).
    """

    rule: str
    phase: str
    metric: str
    severity: int
    message: str
    amateur: float
    delta: float | None
    z_score: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "phase": self.phase,
            "metric": self.metric,
            "severity": self.severity,
            "message": self.message,
            "amateur": self.amateur,
            "delta": self.delta,
            "z_score": self.z_score,
        }


def default_rules_path() -> Path:
    return configs_dir() / "feedback_rules.yaml"


class RuleEngine:
    """Load YAML rules and evaluate them against a :class:`SwingDiff`.

    Usage::

        engine = RuleEngine.from_yaml()
        items = engine.evaluate(swing_diff)

    The engine enforces a single, important invariant: every returned
    :class:`FeedbackItem` is heuristic coaching guidance, not medical or
    clinical advice. That rule is in the docstring rather than the code
    because it is the reader's job, not the engine's.
    """

    def __init__(self, rules: list[FeedbackRule]) -> None:
        self._rules = rules

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> RuleEngine:
        src = Path(path).expanduser().resolve() if path else default_rules_path()
        if not src.is_file():
            raise FileNotFoundError(f"Feedback rules YAML not found: {src}")
        raw = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
        ruleset = FeedbackRuleSet.model_validate(raw)
        return cls(ruleset.rules)

    @property
    def rules(self) -> list[FeedbackRule]:
        return self._rules

    def evaluate(self, diff: SwingDiff) -> list[FeedbackItem]:
        items: list[FeedbackItem] = []
        for rule in self._rules:
            try:
                event = SwingEvent[rule.phase]
            except KeyError:
                _log.warning("Rule %s references unknown phase %s", rule.name, rule.phase)
                continue
            phase = diff.by_event(event)
            if phase is None:
                continue
            delta = phase.by_metric(rule.metric)
            if delta is None or delta.delta is None:
                # Without a cohort delta we cannot evaluate threshold-
                # based rules. Skip silently.
                continue

            signed_delta = delta.delta
            fires = (rule.direction == "above" and signed_delta >= rule.threshold) or (
                rule.direction == "below" and signed_delta <= rule.threshold
            )
            if not fires:
                continue

            try:
                rendered = rule.message_template.format(
                    delta=signed_delta,
                    amateur=delta.amateur,
                )
            except (IndexError, KeyError) as exc:
                _log.warning(
                    "Rule %s has a bad message_template (%s); using raw template.",
                    rule.name,
                    exc,
                )
                rendered = rule.message_template

            items.append(
                FeedbackItem(
                    rule=rule.name,
                    phase=rule.phase,
                    metric=rule.metric,
                    severity=rule.severity,
                    message=rendered,
                    amateur=delta.amateur,
                    delta=signed_delta,
                    z_score=delta.z_score,
                )
            )

        items.sort(key=lambda i: (-i.severity, i.rule))
        return items
