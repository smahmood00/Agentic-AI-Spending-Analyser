"""Shared state type threaded through every node of the analysis pipeline.

LangGraph merges each node's return dict into this TypedDict, so nodes only
need to return the keys they produce or modify.
"""
from __future__ import annotations

from typing import Optional, TypedDict


class PipelineState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    input_path: str          # original upload — PDF or CSV
    csv_path: Optional[str]  # set by extract_node; CSV ready for parsing

    # ── Per-stage outputs ──────────────────────────────────────────────────────
    # Types are kept as Any-compatible to avoid circular imports and to
    # remain compatible with MemorySaver's in-memory (non-JSON) checkpointing.
    parsed: Optional[object]       # src.parser.ParsedStatement
    cat_result: Optional[object]   # src.categorizer.CategorizerResult
    metrics: Optional[dict]
    report: Optional[str]
    verdict: Optional[dict]        # {"approved": bool, "issues": list[str]}

    # ── HITL ───────────────────────────────────────────────────────────────────
    # Populated by categorize_node; consumed by human_review_node.
    # Each item: {description, amount, date_str, reason}
    hitl_pending: list[dict]

    # ── Control flow ───────────────────────────────────────────────────────────
    # Incremented by critique_node when it rejects the report.
    # The conditional edge uses this to allow at most one revision cycle.
    revision_count: int
