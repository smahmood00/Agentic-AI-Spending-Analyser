"""Categorizer agent: rules → cache → batched LLM → HITL.

Hard rule (from project memory): P2P-to-person-name transfers never get a purpose
inferred. They are always `P2P Transfer / Unknown Purpose` until the user labels them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from src.llm_client import LLMClient

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = [
    "P2P Transfer",
    "Cash",
    "Rebate",
    "Credit Card Payment",
    "Card Purchase",
    "Bill / Utility",
    "Food & Dining",
    "Shopping",
    "Subscription",
    "Transport",
    "Income",
    "Rent",
    "Other",
]

PERSON_NAME_RE = re.compile(
    r"^(MR|MRS|MS|MISS|DR|MX)\.?\s+[A-Z][A-Z ]+$", re.IGNORECASE
)


@dataclass
class CategoryLabel:
    category: str
    subcategory: str
    confidence: float
    source: str  # "rule" | "cache" | "llm" | "hitl"


@dataclass
class CategorizerResult:
    labeled: pd.DataFrame  # transactions + category/subcategory/confidence/source
    llm_calls: int = 0
    cache_hits: int = 0
    rule_hits: int = 0
    hitl_prompts: int = 0
    new_cache_entries: int = 0


_PAYME_RE = re.compile(r"\bPAYME\b")


def _rule_match(description: str, ref: str) -> CategoryLabel | None:
    desc = description.upper()
    # Credit-card payment must be checked BEFORE PAYME, because "PAYMENT" contains "PAYME".
    if "HSBC VISA" in desc and "PAYMENT" in desc:
        return CategoryLabel("Credit Card Payment", "HSBC Visa", 1.0, "rule")
    if _PAYME_RE.search(desc):
        return CategoryLabel("P2P Transfer", "PayMe", 0.95, "rule")
    if "ATM WITHDRAWAL" in desc:
        return CategoryLabel("Cash", "ATM", 1.0, "rule")
    if "CASH REBATE" in desc:
        return CategoryLabel("Rebate", "Card Rebate", 1.0, "rule")
    return None


def _is_person_name(description: str) -> bool:
    return bool(PERSON_NAME_RE.match(description.strip()))


def _cache_key(description: str) -> str:
    return description.strip().upper()


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


CATEGORIZER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["labels"],
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["row_id", "category", "subcategory", "confidence"],
                "properties": {
                    "row_id": {"type": "integer"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "subcategory": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            },
        }
    },
}


def _llm_batch_categorize(
    client: LLMClient,
    rows: list[dict],
) -> dict[int, CategoryLabel]:
    if not rows:
        return {}
    system = (
        "You categorize bank transactions. Given a list of transactions, return a label "
        "for each. Use only the allowed categories. Set confidence between 0 and 1. "
        "Never guess the purpose of person-to-person transfers; if unsure, return "
        "category='Other', subcategory='Unknown', confidence below 0.5.\n"
        f"Allowed categories: {CATEGORIES}"
    )
    user = "Transactions to categorize:\n" + json.dumps(rows, indent=2)
    resp = client.complete(
        role="categorizer",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        schema=CATEGORIZER_SCHEMA,
    )
    out: dict[int, CategoryLabel] = {}
    for item in resp.parsed["labels"]:
        out[item["row_id"]] = CategoryLabel(
            category=item["category"],
            subcategory=item["subcategory"],
            confidence=float(item["confidence"]),
            source="llm",
        )
    return out


def _prompt_user(
    description: str,
    amount: float,
    date_str: str,
    reason: str,
) -> CategoryLabel:
    print()
    print("=" * 72)
    print(f"HITL: needs your label  ({reason})")
    print(f"  Date:        {date_str}")
    print(f"  Description: {description}")
    print(f"  Amount:      {amount:+.2f}")
    print("-" * 72)
    print("Allowed categories:")
    for i, c in enumerate(CATEGORIES, 1):
        print(f"  {i:2d}. {c}")
    while True:
        choice = input("Pick number, or type a custom category: ").strip()
        if not choice:
            print("  please enter a value")
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(CATEGORIES):
            category = CATEGORIES[int(choice) - 1]
        else:
            category = choice  # accepted as-is: known label or custom free text
        break
    return CategoryLabel(category=category, subcategory="User Labeled", confidence=1.0, source="hitl")


def categorize(
    transactions: pd.DataFrame,
    client: LLMClient,
    cache_path: str | Path,
    confidence_threshold: float = 0.75,
    prompt_fn: Callable | None = None,
) -> CategorizerResult:
    """Run the three-tier categorization pipeline with HITL.

    prompt_fn is injected for testability; defaults to interactive CLI prompt.
    """
    prompt_fn = prompt_fn or _prompt_user
    cache_path = Path(cache_path)
    cache = _load_cache(cache_path)

    df = transactions.copy()
    labels: dict[int, CategoryLabel] = {}
    p2p_person_rows: set[int] = set()
    result = CategorizerResult(labeled=df)

    # Tier 1: cache → rules → person-name detect.
    # Order matters: cache wins so user-confirmed labels are honored on re-runs.
    unknowns: list[dict] = []
    for _, row in df.iterrows():
        rid = int(row["row_id"])
        desc = str(row["description"])
        ref = str(row["reference"])
        key = _cache_key(desc)

        if key in cache:
            entry = cache[key]
            labels[rid] = CategoryLabel(
                category=entry["category"],
                subcategory=entry["subcategory"],
                confidence=float(entry.get("confidence", 1.0)),
                source="cache",
            )
            result.cache_hits += 1
            continue

        rule_label = _rule_match(desc, ref)
        if rule_label is not None:
            labels[rid] = rule_label
            result.rule_hits += 1
            continue

        if _is_person_name(desc):
            labels[rid] = CategoryLabel("P2P Transfer", "Unknown Purpose", 0.3, "rule")
            p2p_person_rows.add(rid)
            result.rule_hits += 1
            continue

        unknowns.append({
            "row_id": rid,
            "description": desc,
            "amount": float(row["amount"]),
            "reference": ref,
        })

    # Tier 2: batched LLM call for everything still unlabeled.
    # LLM labels meeting the confidence threshold are cached so warm runs skip the LLM.
    if unknowns:
        llm_labels = _llm_batch_categorize(client, unknowns)
        result.llm_calls = 1
        unknowns_by_rid = {u["row_id"]: u for u in unknowns}
        for rid, lbl in llm_labels.items():
            labels[rid] = lbl
            if lbl.confidence >= confidence_threshold:
                desc = unknowns_by_rid[rid]["description"]
                cache[_cache_key(desc)] = {
                    "category": lbl.category,
                    "subcategory": lbl.subcategory,
                    "confidence": lbl.confidence,
                }
                result.new_cache_entries += 1

    # Tier 3: HITL for low-confidence rows and P2P-to-person rows.
    # First occurrence of a description triggers HITL; subsequent ones use the
    # label just written into the cache.
    for _, row in df.iterrows():
        rid = int(row["row_id"])
        lbl = labels.get(rid)
        if lbl is None:
            continue

        needs_hitl = rid in p2p_person_rows or lbl.confidence < confidence_threshold
        if not needs_hitl:
            continue

        key = _cache_key(str(row["description"]))
        if key in cache:
            entry = cache[key]
            labels[rid] = CategoryLabel(
                category=entry["category"],
                subcategory=entry["subcategory"],
                confidence=float(entry.get("confidence", 1.0)),
                source="cache",
            )
            continue

        reason = "P2P to person name" if rid in p2p_person_rows else f"low confidence ({lbl.confidence:.2f})"
        user_label = prompt_fn(
            description=str(row["description"]),
            amount=float(row["amount"]),
            date_str=str(row["date"]),
            reason=reason,
        )
        labels[rid] = user_label
        result.hitl_prompts += 1
        # "pending" source means the UI is collecting labels deferred — don't cache yet.
        # The UI will write to cache after the user submits the form.
        if user_label.source != "pending":
            cache[key] = {
                "category": user_label.category,
                "subcategory": user_label.subcategory,
                "confidence": 1.0,
            }
            result.new_cache_entries += 1

    _save_cache(cache_path, cache)

    df["category"] = df["row_id"].map(lambda r: labels[int(r)].category)
    df["subcategory"] = df["row_id"].map(lambda r: labels[int(r)].subcategory)
    df["confidence"] = df["row_id"].map(lambda r: labels[int(r)].confidence)
    df["label_source"] = df["row_id"].map(lambda r: labels[int(r)].source)

    # Hard-rule assertion: P2P-to-person rows never carry a system-inferred purpose.
    # Only user-confirmed labels (via HITL, or cache from a prior HITL) can override.
    for rid in p2p_person_rows:
        lbl = labels[rid]
        if lbl.source not in ("hitl", "cache"):
            assert lbl.subcategory == "Unknown Purpose", (
                f"hard-rule violation: row {rid} P2P-to-person should be Unknown Purpose"
            )

    result.labeled = df
    return result
