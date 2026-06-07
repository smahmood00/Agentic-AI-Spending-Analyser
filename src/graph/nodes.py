"""LangGraph node functions for the financial analysis pipeline.

Each node receives the full PipelineState and returns a partial dict with only
the keys it produces or modifies. LangGraph merges those updates into state.

Shared resources (LLMClient, app config) are passed via RunnableConfig so
callers can swap them for testing without touching the graph topology.
"""
from __future__ import annotations

from pathlib import Path

from langgraph.types import RunnableConfig, interrupt

from src.analyst import build_metrics, write_metrics
from src.categorizer import (
    CATEGORIES,
    CategoryLabel,
    _cache_key,
    _load_cache,
    _save_cache,
    categorize,
)
from src.critic import review
from src.llm_client import LLMClient
from src.parser import parse_csv
from src.pdf_extractor import extract_statement
from src.reporter import generate_report

from .state import PipelineState

# Project root — two levels up from src/graph/nodes.py
ROOT = Path(__file__).resolve().parent.parent.parent


# ── Config helpers ────────────────────────────────────────────────────────────

def _client(config: RunnableConfig) -> LLMClient:
    return config["configurable"]["llm_client"]


def _cfg(config: RunnableConfig) -> dict:
    return config["configurable"]["app_config"]


# ── Node 1: extract ───────────────────────────────────────────────────────────

def extract_node(state: PipelineState, _config: RunnableConfig) -> dict:
    """Convert a PDF bank statement to CSV. Pass-through if input is already a CSV."""
    path = Path(state["input_path"])
    if path.suffix.lower() == ".pdf":
        csv_path = extract_statement(path, path.with_suffix(".csv"))
    else:
        csv_path = path
    return {"csv_path": str(csv_path)}


# ── Node 2: parse ─────────────────────────────────────────────────────────────

def parse_node(state: PipelineState, config: RunnableConfig) -> dict:
    """Parse the bank statement CSV into a structured DataFrame."""
    parsed = parse_csv(state["csv_path"])
    return {"parsed": parsed}


# ── Node 2: categorize ────────────────────────────────────────────────────────

def categorize_node(state: PipelineState, config: RunnableConfig) -> dict:
    """Run 3-tier categorisation: cache → rules → batched LLM.

    Transactions below the confidence threshold or person-name P2P transfers
    are collected as HITL items rather than blocking here. The human_review
    node handles them via LangGraph's interrupt mechanism.
    """
    app_cfg = _cfg(config)
    pending: list[dict] = []

    def collecting_prompt_fn(
        description: str, amount: float, date_str: str, reason: str
    ) -> CategoryLabel:
        pending.append({
            "description": description,
            "amount": amount,
            "date_str": date_str,
            "reason": reason,
        })
        # Placeholder — human_review_node will replace this with the real label.
        return CategoryLabel("Other", "Pending", 0.0, "pending")

    cat_result = categorize(
        transactions=state["parsed"].transactions,
        client=_client(config),
        cache_path=ROOT / app_cfg["paths"]["merchant_labels"],
        confidence_threshold=app_cfg["thresholds"]["llm_confidence_min"],
        prompt_fn=collecting_prompt_fn,
    )

    # Deduplicate by description — one review prompt per unique merchant.
    seen: set[str] = set()
    deduped = [
        item for item in pending
        if item["description"] not in seen and not seen.add(item["description"])
    ]

    return {"cat_result": cat_result, "hitl_pending": deduped}


# ── Node 3: human_review ──────────────────────────────────────────────────────

def human_review_node(state: PipelineState, config: RunnableConfig) -> dict:
    """Pause the graph and wait for the user to label pending transactions.

    interrupt() serialises hitl_pending to the caller and suspends the graph.
    The graph resumes when the caller provides Command(resume={description: category}).
    On resumption LangGraph re-calls this function; interrupt() immediately
    returns the resume value so label application runs without re-suspending.
    """
    app_cfg = _cfg(config)
    cache_path = ROOT / app_cfg["paths"]["merchant_labels"]

    # Suspend and hand pending transactions to the caller; receive user labels.
    user_labels: dict[str, str] = interrupt(state["hitl_pending"])

    # Apply user labels to the labeled DataFrame and persist them to cache.
    cache = _load_cache(cache_path)
    df = state["cat_result"].labeled

    for description, category in user_labels.items():
        mask = (df["description"] == description) & (df["label_source"] == "pending")
        df.loc[mask, ["category", "subcategory", "confidence", "label_source"]] = (
            category, "User Labeled", 1.0, "hitl"
        )
        cache[_cache_key(description)] = {
            "category": category,
            "subcategory": "User Labeled",
            "confidence": 1.0,
        }

    _save_cache(cache_path, cache)
    state["cat_result"].labeled = df
    return {"cat_result": state["cat_result"], "hitl_pending": []}


# ── Node 4: analyze ───────────────────────────────────────────────────────────

def analyze_node(state: PipelineState, config: RunnableConfig) -> dict:
    """Compute metrics from labeled transactions and write metrics.json."""
    app_cfg = _cfg(config)
    parsed = state["parsed"]
    metrics = build_metrics(
        labeled=state["cat_result"].labeled,
        opening_balance=parsed.opening_balance,
        period_start=parsed.period_start,
        period_end=parsed.period_end,
        recurring_min=app_cfg["thresholds"]["recurring_min_occurrences"],
        anomaly_iqr=app_cfg["thresholds"]["anomaly_iqr_multiplier"],
    )
    write_metrics(metrics, ROOT / app_cfg["paths"]["metrics_out"])
    return {"metrics": metrics}


# ── Node 5: report ────────────────────────────────────────────────────────────

def report_node(state: PipelineState, config: RunnableConfig) -> dict:
    """Generate a 7-section Markdown report from metrics.

    On a revision pass (revision_count > 0) the critic's feedback is forwarded
    to the reporter so it can address the flagged issues.
    """
    feedback = None
    if state["revision_count"] > 0 and state.get("verdict"):
        feedback = state["verdict"].get("issues")

    report = generate_report(_client(config), state["metrics"], critic_feedback=feedback)
    return {"report": report}


# ── Node 6: critique ──────────────────────────────────────────────────────────

def critique_node(state: PipelineState, config: RunnableConfig) -> dict:
    """Review the draft report for grounding violations.

    Increments revision_count when the report is rejected. The conditional
    edge reads this counter to enforce the one-revision-maximum policy.
    """
    verdict = review(_client(config), state["report"], state["metrics"])
    revision_count = state["revision_count"]
    if not verdict["approved"]:
        revision_count += 1
    return {"verdict": verdict, "revision_count": revision_count}
