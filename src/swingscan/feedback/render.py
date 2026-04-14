"""Render :class:`FeedbackItem` lists as plain text or JSON.

Kept separate from :mod:`swingscan.feedback.rules` so the renderer can
be unit-tested without loading any YAML. The caps and ordering live
here, not in the engine, because the CLI and demo UI might choose
different limits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from swingscan.feedback.rules import FeedbackItem

__all__ = ["FeedbackReport", "render_text", "render_json"]


@dataclass(frozen=True, slots=True)
class FeedbackReport:
    items: tuple[FeedbackItem, ...]
    max_items: int = 5

    @property
    def visible(self) -> tuple[FeedbackItem, ...]:
        return self.items[: self.max_items]


def render_text(items: list[FeedbackItem], max_items: int = 5) -> str:
    """Render a bullet-list report with severity markers."""
    if not items:
        return "No feedback items — pose or cohort data were insufficient.\n"
    lines: list[str] = ["SwingScan feedback (heuristic coaching cues):"]
    for item in items[:max_items]:
        marker = "!" * item.severity
        lines.append(f"  [{marker}] {item.phase:>20s} · {item.message}")
    if len(items) > max_items:
        lines.append(f"  ... +{len(items) - max_items} additional cues suppressed")
    lines.append(
        "These are heuristic cues, not medical or clinical advice."
    )
    return "\n".join(lines) + "\n"


def render_json(items: list[FeedbackItem], max_items: int = 5) -> str:
    payload = {
        "format": "swingscan_feedback_v1",
        "items": [i.as_dict() for i in items[:max_items]],
        "total_items": len(items),
        "capped_at": max_items,
        "disclaimer": "Heuristic coaching cues only. Not medical or clinical advice.",
    }
    return json.dumps(payload, indent=2)
