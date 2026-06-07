"""Analyst agent: deterministic pandas tools, emits metrics.json."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, float):
        return round(obj, 2)
    return obj


def aggregate_by_category(df: pd.DataFrame) -> dict:
    out = []
    spend_total = float(-df.loc[df["amount"] < 0, "amount"].sum())
    for cat, sub in df.groupby("category"):
        deposits = float(sub.loc[sub["amount"] > 0, "amount"].sum())
        withdrawals = float(-sub.loc[sub["amount"] < 0, "amount"].sum())
        net = deposits - withdrawals
        share = (withdrawals / spend_total) if spend_total > 0 else 0.0
        out.append({
            "category": cat,
            "deposits": deposits,
            "withdrawals": withdrawals,
            "net": net,
            "share_of_spend": round(share, 4),
            "count": int(len(sub)),
        })
    return sorted(out, key=lambda r: r["withdrawals"], reverse=True)


def top_merchants(df: pd.DataFrame, n: int = 10) -> list[dict]:
    spend = df.loc[df["amount"] < 0].copy()
    spend["abs_amount"] = -spend["amount"]
    grouped = (
        spend.groupby("description")
        .agg(total_spent=("abs_amount", "sum"), count=("amount", "size"))
        .sort_values("total_spent", ascending=False)
        .head(n)
        .reset_index()
    )
    return [
        {
            "merchant": r["description"],
            "total_spent": float(r["total_spent"]),
            "count": int(r["count"]),
        }
        for _, r in grouped.iterrows()
    ]


def cashflow_summary(df: pd.DataFrame, opening_balance: float) -> dict:
    deposits = float(df.loc[df["amount"] > 0, "amount"].sum())
    withdrawals = float(-df.loc[df["amount"] < 0, "amount"].sum())
    net = deposits - withdrawals
    closing = opening_balance + net
    savings_rate = (net / deposits) if deposits > 0 else 0.0
    return {
        "opening_balance": opening_balance,
        "closing_balance": closing,
        "total_deposits": deposits,
        "total_withdrawals": withdrawals,
        "net_cashflow": net,
        "savings_rate": round(savings_rate, 4),
    }


def detect_anomalies(df: pd.DataFrame, iqr_mult: float = 1.5) -> list[dict]:
    spend = df.loc[df["amount"] < 0].copy()
    spend["abs_amount"] = -spend["amount"]
    if spend.empty:
        return []
    q1 = spend["abs_amount"].quantile(0.25)
    q3 = spend["abs_amount"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + iqr_mult * iqr
    outliers = spend.loc[spend["abs_amount"] > upper].sort_values("abs_amount", ascending=False)
    return [
        {
            "date": r["date"],
            "merchant": r["description"],
            "category": r["category"],
            "amount": float(r["abs_amount"]),
            "threshold": float(upper),
        }
        for _, r in outliers.iterrows()
    ]


def detect_recurring(df: pd.DataFrame, min_occurrences: int = 3) -> list[dict]:
    spend = df.loc[df["amount"] < 0].copy()
    if spend.empty:
        return []
    spend["month"] = spend["date"].apply(lambda d: (d.year, d.month))
    recurring: list[dict] = []
    for merchant, sub in spend.groupby("description"):
        months = sub["month"].nunique()
        if months >= min_occurrences:
            recurring.append({
                "merchant": merchant,
                "occurrences": int(len(sub)),
                "distinct_months": int(months),
                "total_spent": float(-sub["amount"].sum()),
            })
    return sorted(recurring, key=lambda r: r["total_spent"], reverse=True)


def needs_review(df: pd.DataFrame) -> list[dict]:
    flagged = df.loc[df["subcategory"] == "Unknown Purpose"]
    return [
        {
            "date": r["date"],
            "description": r["description"],
            "amount": float(r["amount"]),
            "category": r["category"],
        }
        for _, r in flagged.iterrows()
    ]


def build_metrics(
    labeled: pd.DataFrame,
    opening_balance: float,
    period_start,
    period_end,
    recurring_min: int = 3,
    anomaly_iqr: float = 1.5,
) -> dict:
    metrics = {
        "period": {
            "start": period_start,
            "end": period_end,
            "days": (period_end - period_start).days + 1,
        },
        "cashflow": cashflow_summary(labeled, opening_balance),
        "by_category": aggregate_by_category(labeled),
        "top_merchants": top_merchants(labeled),
        "anomalies": detect_anomalies(labeled, iqr_mult=anomaly_iqr),
        "recurring": detect_recurring(labeled, min_occurrences=recurring_min),
        "needs_review": needs_review(labeled),
        "transaction_count": int(len(labeled)),
    }
    return _to_native(metrics)


def write_metrics(metrics: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
