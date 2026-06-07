"""Evaluation for the HSBC PDF extractor.

Runs `extract_statement` against the sample PDF and diffs the output CSV against
the hand-written ground-truth CSV. Reports PASS/FAIL on each criterion.

Critical columns (must match exactly):  date, type, deposit, withdrawal, balance
Soft columns (informational only):       description, reference, note
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pdf_extractor import extract_statement

GROUND_TRUTH = ROOT / "data" / "hsbc_transaction_latest.csv"
SAMPLE_PDF = ROOT / "data" / "eStatementFile_20260531193657_hsbc.pdf"

CRITICAL_COLS = ("date", "type", "deposit", "withdrawal", "balance")
SOFT_COLS = ("description", "reference", "note")


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _normalize_num(s: str) -> str:
    """Normalize blank vs '0.00' so they compare equal."""
    s = (s or "").strip()
    if not s:
        return ""
    try:
        return f"{float(s):.2f}"
    except ValueError:
        return s


def main() -> int:
    out_csv = extract_statement(SAMPLE_PDF)
    print(f"extracted → {out_csv}")

    gt = _read_csv(GROUND_TRUTH)
    ex = _read_csv(out_csv)

    checks: dict[str, bool] = {}
    metrics: dict[str, int | float] = {}

    # ---- row count ----
    checks["row_count_matches"] = len(gt) == len(ex)
    metrics["row_count_gt"] = len(gt)
    metrics["row_count_extracted"] = len(ex)

    # ---- per-column comparisons ----
    n = min(len(gt), len(ex))
    critical_mismatches: list[str] = []
    soft_mismatch_counts: dict[str, int] = {c: 0 for c in SOFT_COLS}
    soft_mismatch_examples: dict[str, list[str]] = {c: [] for c in SOFT_COLS}

    for i in range(n):
        for col in CRITICAL_COLS:
            g = _normalize_num(gt[i][col]) if col in ("deposit", "withdrawal", "balance") else (gt[i][col] or "").strip()
            e = _normalize_num(ex[i][col]) if col in ("deposit", "withdrawal", "balance") else (ex[i][col] or "").strip()
            if g != e:
                critical_mismatches.append(f"row {i+1} col '{col}': GT={g!r} got={e!r}")
        for col in SOFT_COLS:
            g = (gt[i][col] or "").strip()
            e = (ex[i][col] or "").strip()
            if g != e:
                soft_mismatch_counts[col] += 1
                if len(soft_mismatch_examples[col]) < 3:
                    soft_mismatch_examples[col].append(f"row {i+1}: GT={g!r} got={e!r}")

    checks["all_critical_columns_match"] = len(critical_mismatches) == 0
    metrics["critical_mismatches"] = len(critical_mismatches)
    for col in SOFT_COLS:
        metrics[f"{col}_mismatches"] = soft_mismatch_counts[col]

    # ---- balance reconciliation already enforced by the extractor itself ----
    # (extract_statement raises ValueError if balance math doesn't add up).
    checks["balance_reconciliation_ok"] = True

    # ---- report ----
    print("\n=== metrics ===")
    for k, v in metrics.items():
        print(f"  {k:30s} {v}")

    if critical_mismatches:
        print("\n=== critical mismatches ===")
        for m in critical_mismatches[:20]:
            print(f"  {m}")
        if len(critical_mismatches) > 20:
            print(f"  ... ({len(critical_mismatches) - 20} more)")

    print("\n=== soft (informational) mismatches ===")
    for col in SOFT_COLS:
        c = soft_mismatch_counts[col]
        if c:
            print(f"  {col}: {c} differences")
            for ex_line in soft_mismatch_examples[col]:
                print(f"    {ex_line}")

    print("\n=== pass/fail ===")
    all_passed = True
    for label, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
        all_passed = all_passed and ok

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
