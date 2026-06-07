"""CSV → cleaned pandas DataFrame for downstream agents."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# Hang Seng embeds "(DR=Debit)" / "(CR=Credit)" labels in the description column.
_HANGSENG_DR_CR_RE = re.compile(r"\(\s*[DC]R=[^)]+\)\s*", re.IGNORECASE)

REQUIRED_COLUMNS = [
    "date",
    "description",
    "reference",
    "type",
    "deposit",
    "withdrawal",
    "balance",
    "note",
]

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


@dataclass
class ParsedStatement:
    transactions: pd.DataFrame
    period_start: date
    period_end: date
    opening_balance: float


def _infer_dates(raw_dates: list[str], today: date) -> list[date]:
    """Convert '11 Apr' strings to absolute dates.

    Anchor: the first row's year is chosen so that no parsed date exceeds today.
    Walk forward: if month decreases vs prior row, increment year (Dec→Jan rollover).
    """
    parsed: list[tuple[int, int]] = []
    for s in raw_dates:
        s = s.strip()
        day_str, mon_str = s.split()
        parsed.append((int(day_str), MONTH_MAP[mon_str]))

    first_day, first_month = parsed[0]
    year = today.year if first_month <= today.month else today.year - 1
    candidate = date(year, first_month, first_day)
    if candidate > today:
        year -= 1

    out: list[date] = []
    prev_month = parsed[0][1]
    for day, month in parsed:
        if month < prev_month:
            year += 1
        out.append(date(year, month, day))
        prev_month = month
    return out


def parse_csv(path: str | Path, today: date | None = None) -> ParsedStatement:
    today = today or date.today()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["description"] = df["description"].str.replace(_HANGSENG_DR_CR_RE, "", regex=True).str.strip()
    df["date"] = _infer_dates(df["date"].tolist(), today)

    for col in ("deposit", "withdrawal", "balance"):
        df[col] = pd.to_numeric(df[col].replace("", "0"), errors="coerce").fillna(0.0)

    bf_mask = df["type"].str.lower() == "balance"
    opening_balance = float(df.loc[bf_mask, "balance"].iloc[0]) if bf_mask.any() else 0.0

    non_txn_mask = df["type"].str.lower().isin(["balance", "closingbalance"])
    df = df.loc[~non_txn_mask].reset_index(drop=True)
    df["amount"] = df["deposit"] - df["withdrawal"]

    running = opening_balance + df["amount"].cumsum()
    df["balance"] = running

    df.insert(0, "row_id", range(1, len(df) + 1))

    return ParsedStatement(
        transactions=df,
        period_start=df["date"].min(),
        period_end=df["date"].max(),
        opening_balance=opening_balance,
    )


if __name__ == "__main__":
    import sys
    p = parse_csv(sys.argv[1])
    print(f"Period: {p.period_start} → {p.period_end}")
    print(f"Opening balance: {p.opening_balance:.2f}")
    print(f"Rows: {len(p.transactions)}")
    print(p.transactions.head(10).to_string())
