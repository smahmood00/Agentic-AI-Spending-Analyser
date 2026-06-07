"""CLI entry point for the AI Financial Analyst pipeline.

Runs the LangGraph pipeline, streaming node-level progress to stdout and
handling HITL interrupts interactively via the terminal.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from langgraph.types import Command

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.categorizer import CATEGORIES
from src.graph import build_pipeline
from src.graph.state import PipelineState
from src.llm_client import LLMClient

# Human-readable label for each node shown in progress output.
_NODE_LABELS = {
    "parse":        "[1/6] Parsing statement",
    "categorize":   "[2/6] Categorising transactions",
    "human_review": "[3/6] Human review",
    "analyze":      "[4/6] Computing metrics",
    "report":       "[5/6] Generating report",
    "critique":     "[6/6] Critic review",
}


def main() -> int:
    args = _parse_args()

    with open(args.config) as f:
        app_cfg = yaml.safe_load(f)

    client = LLMClient(config_path=args.config)
    pipeline = build_pipeline()

    thread_config = {
        "configurable": {
            "thread_id": "cli-run",
            "llm_client": client,
            "app_config": app_cfg,
        }
    }

    initial_state = PipelineState(
        csv_path=args.csv_path,
        parsed=None,
        cat_result=None,
        hitl_pending=[],
        metrics=None,
        report=None,
        verdict=None,
        revision_count=0,
    )

    _stream(pipeline, initial_state, thread_config, app_cfg)
    return 0


# ── Streaming runner ──────────────────────────────────────────────────────────

def _stream(pipeline, state_or_command, thread_config: dict, app_cfg: dict) -> None:
    """Stream pipeline execution, printing progress and handling HITL."""
    for chunk in pipeline.stream(state_or_command, config=thread_config, stream_mode="updates"):
        for node_name in chunk:
            print(f"  {_NODE_LABELS.get(node_name, node_name)} ... done")

    # Check whether the pipeline paused for human review.
    snapshot = pipeline.get_state(thread_config)
    if snapshot.next:
        pending: list[dict] = snapshot.tasks[0].interrupts[0].value
        user_labels = _collect_labels_from_terminal(pending)
        _stream(pipeline, Command(resume=user_labels), thread_config, app_cfg)
        return

    # Pipeline finished — write outputs.
    final = pipeline.get_state(thread_config).values
    report_path = ROOT / app_cfg["paths"]["report_out"]
    report_path.write_text(final["report"], encoding="utf-8")
    print(f"\n  Report written to {report_path}")

    _print_cost_summary(ROOT / app_cfg["paths"]["llm_call_log"])


# ── HITL: terminal prompt ─────────────────────────────────────────────────────

def _collect_labels_from_terminal(pending: list[dict]) -> dict[str, str]:
    """Prompt the user to label each pending transaction via stdin."""
    user_labels: dict[str, str] = {}
    for txn in pending:
        print()
        print("=" * 72)
        print(f"  Needs your label  ({txn['reason']})")
        print(f"  Date:        {txn['date_str']}")
        print(f"  Description: {txn['description']}")
        print(f"  Amount:      {txn['amount']:+.2f}")
        print("-" * 72)
        for i, cat in enumerate(CATEGORIES, 1):
            print(f"  {i:2d}. {cat}")
        while True:
            choice = input("Pick number, or type a custom category: ").strip()
            if not choice:
                print("  Please enter a value.")
                continue
            if choice.isdigit() and 1 <= int(choice) <= len(CATEGORIES):
                user_labels[txn["description"]] = CATEGORIES[int(choice) - 1]
            else:
                user_labels[txn["description"]] = choice
            break
    return user_labels


# ── Cost summary ──────────────────────────────────────────────────────────────

def _print_cost_summary(log_path: Path) -> None:
    if not log_path.exists():
        return
    counts: Counter = Counter()
    in_tok: Counter = Counter()
    out_tok: Counter = Counter()
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            role = entry["role"]
            counts[role] += 1
            in_tok[role] += entry["input_tokens"]
            out_tok[role] += entry["output_tokens"]
    print("\n--- LLM call summary (cumulative across all runs) ---")
    for role in sorted(counts):
        print(f"  {role:12s}  calls={counts[role]:3d}  "
              f"in={in_tok[role]:6d} tokens  out={out_tok[role]:6d} tokens")


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Financial Analyst")
    parser.add_argument("csv_path", help="Path to bank statement CSV")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
