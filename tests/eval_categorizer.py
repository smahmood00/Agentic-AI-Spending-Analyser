"""Evaluation for the categorizer agent.

Metrics (per plan):
- Rule-tier precision ≥ 0.98
- Overall accuracy (category) vs gold ≥ 0.90
- HITL trigger recall = 1.0 for P2P-to-person rows
- Cache hit rate on re-run ≥ 0.95
- Batching check: LLM call count == 1 on a cold run with N unknowns

Runs against a mocked HITL prompt (no terminal input needed) and a real LLM call
(uses the configured OpenRouter model). To skip the live LLM and use a stub,
pass --offline.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.categorizer import (
    CATEGORIES,
    CategoryLabel,
    categorize,
)
from src.llm_client import LLMClient


GOLD_PATH = ROOT / "tests" / "fixtures" / "categorizer_gold.json"


class StubLLMClient:
    """Offline stand-in for the real LLM client — deterministic labels for the gold set."""

    LABELS = {
        "MAA INTERNATIONAL": ("Shopping", "Retail", 0.8),
        "CAPCUT": ("Subscription", "Software", 0.92),
        "ELEVENLABS.IO": ("Subscription", "AI Service", 0.95),
        "LUNG FUNG MALL": ("Shopping", "Mall", 0.85),
        "KYLE NGUYEN": ("Food & Dining", "Unknown Vendor", 0.55),
    }

    def __init__(self):
        self.call_count = 0

    def complete(self, role, messages, schema=None, temperature=0.0):
        self.call_count += 1
        user_msg = messages[-1]["content"]
        rows = json.loads(user_msg.split("Transactions to categorize:\n", 1)[1])
        out = []
        for row in rows:
            cat, sub, conf = self.LABELS.get(
                row["description"].upper(), ("Other", "Unknown", 0.4)
            )
            out.append({
                "row_id": row["row_id"],
                "category": cat,
                "subcategory": sub,
                "confidence": conf,
            })
        from src.llm_client import LLMResponse
        body = {"labels": out}
        return LLMResponse(
            content=json.dumps(body),
            parsed=body,
            model="stub",
            input_tokens=len(user_msg) // 4,
            output_tokens=len(json.dumps(body)) // 4,
            latency_ms=0,
        )


def _build_df(gold: dict) -> pd.DataFrame:
    rows = []
    for t in gold["transactions"]:
        rows.append({
            "row_id": t["row_id"],
            "date": pd.Timestamp(t["date"]).date(),
            "description": t["description"],
            "reference": t["reference"],
            "amount": t["amount"],
            "deposit": max(t["amount"], 0.0),
            "withdrawal": max(-t["amount"], 0.0),
            "balance": 0.0,
            "type": "Deposit" if t["amount"] > 0 else "Withdrawal",
            "note": "",
        })
    return pd.DataFrame(rows)


def _mock_prompt_fn(label_choice="P2P Transfer", subcategory="Self Transfer"):
    calls = []

    def fn(*, description, amount, date_str, reason):
        calls.append({"description": description, "reason": reason})
        return CategoryLabel(label_choice, subcategory, 1.0, "hitl")

    fn.calls = calls
    return fn


def evaluate(use_real_llm: bool) -> dict:
    with open(GOLD_PATH) as f:
        gold = json.load(f)
    df = _build_df(gold)

    if use_real_llm:
        client = LLMClient()
    else:
        client = StubLLMClient()

    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "merchant_labels.json"

        # Cold run
        prompt_fn = _mock_prompt_fn()
        result1 = categorize(df, client, cache_path, confidence_threshold=0.75, prompt_fn=prompt_fn)
        cold_labeled = result1.labeled
        cold_llm_calls = getattr(client, "call_count", result1.llm_calls)
        hitl_descs = {c["description"] for c in prompt_fn.calls}

        # Warm run (cache populated)
        prompt_fn2 = _mock_prompt_fn()
        if use_real_llm:
            client = LLMClient()  # fresh — separate count
        else:
            client = StubLLMClient()
        result2 = categorize(df, client, cache_path, confidence_threshold=0.75, prompt_fn=prompt_fn2)
        warm_cache_hits = result2.cache_hits
        warm_llm_calls = getattr(client, "call_count", result2.llm_calls)

    # ---- metrics ----
    by_rid = {t["row_id"]: t for t in gold["transactions"]}
    rule_fired_correct = rule_fired_total = 0
    overall_correct = overall_total = 0

    for _, row in cold_labeled.iterrows():
        rid = int(row["row_id"])
        gold_t = by_rid.get(rid)
        if gold_t is None:
            continue
        exp = gold_t["expect"]
        overall_total += 1

        if "category" not in exp or row["category"] == exp["category"]:
            overall_correct += 1

        # Rule precision: when the rule tier fires, is its category correct?
        if row["label_source"] == "rule":
            rule_fired_total += 1
            if "category" not in exp or row["category"] == exp["category"]:
                rule_fired_correct += 1

    # HITL recall: every hitl_required gold description must appear in the
    # set of descriptions the prompt was invoked for during the cold run.
    required_descs = {by_rid[t["row_id"]]["description"] for t in gold["transactions"] if t.get("hitl_required")}
    hitl_recall = 1.0 if not required_descs else (
        1.0 if required_descs.issubset(hitl_descs) else 0.0
    )

    rule_precision = (rule_fired_correct / rule_fired_total) if rule_fired_total else 1.0
    overall_acc = overall_correct / overall_total if overall_total else 0.0

    return {
        "rule_precision": round(rule_precision, 3),
        "overall_accuracy": round(overall_acc, 3),
        "hitl_recall_p2p_person": hitl_recall,
        "llm_calls_cold": cold_llm_calls,
        "llm_calls_warm": warm_llm_calls,
        "hitl_prompts_cold": len(prompt_fn.calls),
        "warm_cache_hits": warm_cache_hits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="Use stub LLM instead of OpenRouter")
    args = ap.parse_args()

    m = evaluate(use_real_llm=not args.offline)
    print("\n=== Categorizer evaluation ===")
    for k, v in m.items():
        print(f"  {k:30s} {v}")

    checks = {
        "rule_precision >= 0.98":        m["rule_precision"] >= 0.98,
        "overall_accuracy >= 0.90":      m["overall_accuracy"] >= 0.90,
        "hitl_recall_p2p_person == 1.0": m["hitl_recall_p2p_person"] == 1.0,
        "llm_calls_cold == 1":           m["llm_calls_cold"] == 1,
        "llm_calls_warm == 0":           m["llm_calls_warm"] == 0,
    }
    print("\n=== Pass/fail ===")
    all_passed = True
    for label, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
        all_passed = all_passed and ok
    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
