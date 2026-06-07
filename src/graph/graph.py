"""Assembles and compiles the LangGraph financial analysis pipeline.

Graph topology
──────────────
                          ┌── human_review ──┐
parse → categorize ───────┤                  ├── analyze → report → critique ──┬── END
                          └──────────────────┘                                 │
                               (skipped if                           (revision) └── report
                               no HITL needed)

Nodes
─────
  parse         Deterministic CSV parsing
  categorize    Cache → rules → batched LLM; collects HITL items as a side-output
  human_review  Interrupts for human labels (skipped when cache/rules cover all)
  analyze       Deterministic pandas aggregations → metrics.json
  report        LLM drafts a 7-section Markdown report from metrics
  critique      LLM reviews the draft; rejects on grounding violations

Edges
─────
  parse → categorize                    always
  categorize → human_review | analyze   conditional (route_after_categorize)
  human_review → analyze                always
  analyze → report                      always
  report → critique                     always
  critique → report | END               conditional (route_after_critique, max 1 revision)
"""
from __future__ import annotations

import pickle

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .edges import route_after_categorize, route_after_critique
from .nodes import (
    analyze_node,
    categorize_node,
    critique_node,
    human_review_node,
    parse_node,
    report_node,
)
from .state import PipelineState


class _PickleSerde:
    """Pickle-based serialiser for MemorySaver.

    LangGraph's default msgpack serialiser cannot handle pandas DataFrames or
    custom dataclasses stored in PipelineState. Pickle supports arbitrary Python
    objects and is safe here because state never leaves the process.
    """

    def dumps_typed(self, obj: object) -> tuple[str, bytes]:
        return "pickle", pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def loads_typed(self, data: tuple[str, bytes]) -> object:
        _, b = data
        return pickle.loads(b)


def build_pipeline(checkpointer=None):
    """Assemble and compile the analysis pipeline as a LangGraph StateGraph.

    Args:
        checkpointer: LangGraph checkpointer used for state persistence and
                      HITL resumption. Defaults to MemorySaver (in-memory).
                      Pass a SqliteSaver for durable cross-process persistence.

    Returns:
        A compiled LangGraph graph ready to invoke or stream.
    """
    builder = StateGraph(PipelineState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    builder.add_node("parse",        parse_node)
    builder.add_node("categorize",   categorize_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("analyze",      analyze_node)
    builder.add_node("report",       report_node)
    builder.add_node("critique",     critique_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    builder.set_entry_point("parse")
    builder.add_edge("parse",        "categorize")
    builder.add_conditional_edges("categorize",  route_after_categorize)
    builder.add_edge("human_review", "analyze")
    builder.add_edge("analyze",      "report")
    builder.add_edge("report",       "critique")
    builder.add_conditional_edges("critique",    route_after_critique)

    return builder.compile(checkpointer=checkpointer or MemorySaver(serde=_PickleSerde()))
