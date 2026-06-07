"""Conditional routing functions for the analysis pipeline graph.

These are pure functions — they inspect state and return the name of the next
node (or END). No side effects, no LLM calls.
"""
from __future__ import annotations

from langgraph.graph import END

from .state import PipelineState


def route_after_categorize(state: PipelineState) -> str:
    """Route to human_review if any transactions need labelling, else skip to analyze."""
    return "human_review" if state["hitl_pending"] else "analyze"


def route_after_critique(state: PipelineState) -> str:
    """Allow at most one revision cycle.

    critique_node increments revision_count each time it rejects, so:
      - revision_count == 0 → first pass, no rejection yet
      - revision_count == 1 → first rejection, route back to report
      - revision_count == 2 → second rejection or approval after revision → stop
    """
    if not state["verdict"]["approved"] and state["revision_count"] <= 1:
        return "report"
    return END
