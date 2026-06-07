"""HSBC eStatement PDF → CSV (deterministic, no LLM, PyMuPDF + word bboxes).

Approach:
1. For each page, get word bboxes via `page.get_text("words")`.
2. Group words into visual rows by rounding y0 to the nearest integer
   (PyMuPDF's own `line_no` is not row-aligned for this layout).
3. Find the header row — the one whose word set covers
   {Date, Transaction, Details, Deposit, Withdrawal, Balance}. Robust to any
   vertical-position shift between statements.
4. Read column x0 positions off that header. Column boundaries for the
   Deposit / Withdrawal / Balance amount columns are the midpoints between
   adjacent header x0 values.
5. For each row below the header until a footer marker, classify each word
   by its x0 into date | description | deposit | withdrawal | balance.
6. Run a state machine over the row stream: a row carrying a deposit or
   withdrawal amount closes one Transaction; preceding rows accumulate as
   description / reference for that Transaction.
7. Validate balance reconciliation (raise on mismatch > HKD 0.01).
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

# ── Shared (HSBC + Hang Seng) ─────────────────────────────────────────────────
HEADER_TOKENS = {"Date", "Transaction", "Details", "Deposit", "Withdrawal", "Balance"}
DATE_RE = re.compile(r"^\d{1,2} [A-Z][a-z]{2}$")
BF_BALANCE_RE = re.compile(r"^B/F BALANCE\b", re.IGNORECASE)
CF_BALANCE_RE = re.compile(r"^C/F BALANCE\b", re.IGNORECASE)

# Stop reading transaction rows when any of these lines appear.
FOOTER_MARKERS = (
    "Total Relationship Balance",   # HSBC
    "Thank you for choosing HSBC",  # HSBC
    "Important Notice",             # HSBC
    "Transaction Summary",          # Hang Seng
)

# ── HSBC reference codes ──────────────────────────────────────────────────────
# Multi-line transactions print a bank reference on a separate visual line.
# These patterns identify that line so it lands in the 'reference' column
# rather than being merged into the description.
HSBC_REFERENCE_RE = re.compile(
    r"(T\d+\w*\(\d{2}[A-Z]{3}\d{2}\))"            # e.g. T12345ABC(01JAN25)
    r"|(HC\d+\s+\d{1,2}[A-Z]{3})"                  # e.g. HC987654 01JAN
    r"|(POS MDC\s*\(\d{2}[A-Z]{3}\d{2}\)P?)"       # e.g. POS MDC(01JAN25)
    r"|(MDC P\s+\(\d{2}[A-Z]{3}\d{2}\))"           # e.g. MDC P (01JAN25)
    r"|(N\d+\(\d{2}[A-Z]{3}\d{2}\))"               # e.g. N12345(01JAN25)
    r"|(ATM WITHDRAWAL\s*\(\d{2}[A-Z]{3}\d{2}\))"  # e.g. ATM WITHDRAWAL(01JAN25)
)

# ── Hang Seng reference codes ─────────────────────────────────────────────────
# Each transaction prints an HD reference on the line below the payee name.
HANGSENG_REFERENCE_RE = re.compile(
    r"HD\d+\s+\d{1,2}[A-Z]{3}"  # e.g. HD12630392630864 03MAR
)

# Hang Seng prints "( DR=Debit )" / "( CR=Credit )" as column labels inside the
# transaction description area. Strip them so they don't pollute merchant names.
HANGSENG_DR_CR_RE = re.compile(r"\(\s*[DC]R=[^)]+\)\s*", re.IGNORECASE)

CSV_COLUMNS = ["date", "description", "reference", "type", "deposit", "withdrawal", "balance", "note"]


@dataclass
class Row:
    """One visual row from the PDF transaction area."""
    date: str = ""
    text: str = ""
    deposit: float | None = None
    withdrawal: float | None = None
    balance: float | None = None


@dataclass
class Transaction:
    date: str
    description: str
    reference: str = ""
    type: str = ""
    deposit: float | None = None
    withdrawal: float | None = None
    balance: float | None = None
    note: str = ""


# ----------------------------------------------------------------------------
# header detection + column boundaries
# ----------------------------------------------------------------------------

@dataclass
class ColumnX:
    """x-coordinates read off the header row. Used to classify words into columns."""
    date_x1: float           # right edge of 'Date' word
    desc_x0: float           # x0 of 'Transaction' (description column start)
    deposit_x0: float        # x0 of 'Deposit'
    withdrawal_x0: float     # x0 of 'Withdrawal'
    balance_x0: float        # x0 of 'Balance'

    @property
    def date_right(self) -> float:
        # Date words end well before description starts. Midpoint between
        # the 'Date' header's right edge and the description column start.
        return (self.date_x1 + self.desc_x0) / 2


def _group_by_visual_row(words: list[tuple]) -> dict[int, list[tuple]]:
    """Group words into visual rows by rounding y0 to the nearest integer."""
    rows: dict[int, list[tuple]] = defaultdict(list)
    for w in words:
        y_bucket = int(round(w[1]))
        rows[y_bucket].append(w)
    return rows


def _find_header(rows: dict[int, list[tuple]]) -> tuple[int, ColumnX] | None:
    """Return (header_y_bucket, ColumnX) for the row that contains all header tokens."""
    for y in sorted(rows):
        words_on_line = rows[y]
        texts = {w[4] for w in words_on_line}
        if HEADER_TOKENS.issubset(texts):
            by_text = {w[4]: w for w in words_on_line}
            return y, ColumnX(
                date_x1=by_text["Date"][2],
                desc_x0=by_text["Transaction"][0],
                deposit_x0=by_text["Deposit"][0],
                withdrawal_x0=by_text["Withdrawal"][0],
                balance_x0=by_text["Balance"][0],
            )
    return None


# ----------------------------------------------------------------------------
# row parsing — assign each word in a visual line to a column by its x-position
# ----------------------------------------------------------------------------

def _parse_amount(s: str) -> float | None:
    s = s.strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _classify_word_to_column(word_text: str, x0: float, x1: float, cols: ColumnX) -> str:
    """Return 'date' | 'desc' | 'deposit' | 'withdrawal' | 'balance'.

    Numeric words in the amount columns are right-aligned, so they always END
    before the next column's header begins — even when the number is wide
    enough to extend leftward past its own column header's x0. Comparing the
    word's right edge (x1) against the next column's x0 is therefore robust
    to column-width and amount-alignment differences between bank layouts.
    """
    is_num = _parse_amount(word_text) is not None
    if x0 < cols.date_right:
        return "date"
    if not is_num:
        return "desc"
    if x1 < cols.deposit_x0 - 1:
        return "desc"             # numeric word still in description area (e.g. account no.)
    if x1 < cols.withdrawal_x0:
        return "deposit"
    if x1 < cols.balance_x0:
        return "withdrawal"
    return "balance"


def _parse_visual_row(words: list[tuple], cols: ColumnX) -> Row:
    words_sorted = sorted(words, key=lambda w: w[0])
    date_words: list[str] = []
    desc_words: list[str] = []
    deposit = withdrawal = balance = None

    for w in words_sorted:
        x0 = w[0]
        x1 = w[2]
        text = w[4]
        col = _classify_word_to_column(text, x0, x1, cols)
        if col == "date":
            date_words.append(text)
        elif col == "desc":
            desc_words.append(text)
        elif col == "deposit":
            deposit = _parse_amount(text)
        elif col == "withdrawal":
            withdrawal = _parse_amount(text)
        elif col == "balance":
            balance = _parse_amount(text)

    date_str = " ".join(date_words).strip()
    return Row(
        date=date_str if DATE_RE.match(date_str) else "",
        text=" ".join(desc_words).strip(),
        deposit=deposit,
        withdrawal=withdrawal,
        balance=balance,
    )


def _extract_page_rows(page) -> list[Row]:
    words = page.get_text("words")
    visual_rows = _group_by_visual_row(words)
    found = _find_header(visual_rows)
    if found is None:
        return []
    header_y, cols = found

    out: list[Row] = []
    for y in sorted(visual_rows):
        if y <= header_y:
            continue
        line_words = visual_rows[y]
        line_text = " ".join(w[4] for w in sorted(line_words, key=lambda w: w[0]))
        if any(m in line_text for m in FOOTER_MARKERS):
            break
        if any(skip in line_text for skip in ("HSBC One", "Page", "IPSSTM", "Branch")):
            continue
        row = _parse_visual_row(line_words, cols)
        if not (row.date or row.text or row.deposit is not None
                or row.withdrawal is not None or row.balance is not None):
            continue
        out.append(row)
    return out


# ----------------------------------------------------------------------------
# multi-line state machine — fold rows into Transactions
# ----------------------------------------------------------------------------

def _looks_like_reference(text: str) -> bool:
    return bool(HSBC_REFERENCE_RE.search(text) or HANGSENG_REFERENCE_RE.search(text))


def _classify_rows(rows: list[Row]) -> list[Transaction]:
    """An amount-bearing row closes a Transaction; prior rows accumulate as desc/ref.

    Detection of single-line vs multi-line is implicit:
      - single-line:  one row carries text AND an amount   → close immediately
      - multi-line:   prior rows carry text only; next row carries the amount → close
    """
    txns: list[Transaction] = []
    current_date = ""
    buffered_text: list[str] = []
    pending_pos_mdc_ref = ""  # POS MDC line shared between a rebate row and the next merchant row

    for row in rows:
        if row.date:
            # New date — clear shared-reference state so refs don't leak across dates.
            if row.date != current_date:
                pending_pos_mdc_ref = ""
            current_date = row.date

        # B/F BALANCE — opening balance row, not a transaction.
        if row.text and BF_BALANCE_RE.search(row.text):
            txns.append(Transaction(
                date=current_date,
                description="B/F BALANCE",
                type="Balance",
                balance=row.balance,
            ))
            buffered_text.clear()
            continue

        # C/F BALANCE — closing balance row, not a transaction.
        if row.text and CF_BALANCE_RE.search(row.text):
            txns.append(Transaction(
                date=current_date,
                description="C/F BALANCE",
                type="ClosingBalance",
                balance=row.balance,
            ))
            buffered_text.clear()
            continue

        if row.text:
            buffered_text.append(row.text)

        if row.deposit is not None or row.withdrawal is not None:
            desc_lines = [t for t in buffered_text if not _looks_like_reference(t)]
            ref_lines = [t for t in buffered_text if _looks_like_reference(t)]

            description = HANGSENG_DR_CR_RE.sub("", " ".join(desc_lines)).strip()
            reference = " ".join(ref_lines).strip() or pending_pos_mdc_ref

            # ATM WITHDRAWAL: description + reference combined in one row.
            if description.startswith("ATM WITHDRAWAL") and not reference:
                m = HSBC_REFERENCE_RE.search(description)
                if m:
                    reference = m.group(0).strip()
                    description = "ATM WITHDRAWAL"
            # When the only thing buffered is an ATM-style ref line, description is empty
            # but ref looks like "ATM WITHDRAWAL (...)" — split it.
            if not description and reference.startswith("ATM WITHDRAWAL"):
                description = "ATM WITHDRAWAL"

            txns.append(Transaction(
                date=current_date,
                description=description,
                reference=reference,
                type="Deposit" if row.deposit is not None else "Withdrawal",
                deposit=row.deposit,
                withdrawal=row.withdrawal,
                balance=row.balance,
            ))

            pending_pos_mdc_ref = reference if reference and "POS MDC" in reference else ""
            buffered_text.clear()
            continue

        # Standalone POS MDC line — remember as shared reference for next merchant row,
        # AND retroactively patch the previous transaction on the same date if it
        # still has no reference (rebate row that closed just before this POS MDC line).
        if row.text and "POS MDC" in row.text:
            pending_pos_mdc_ref = row.text.strip()
            if txns and txns[-1].date == current_date and not txns[-1].reference:
                txns[-1].reference = pending_pos_mdc_ref

    return txns


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------

def _validate(transactions: list[Transaction]) -> None:
    opening_row = next((t for t in transactions if t.type == "Balance"), None)
    if opening_row is None or opening_row.balance is None:
        raise ValueError("No opening (B/F) balance row found")
    opening = opening_row.balance

    txn_rows = [t for t in transactions if t.type in ("Deposit", "Withdrawal")]
    if not txn_rows:
        raise ValueError("No transactions extracted")

    running = opening
    last_printed = opening
    for t in txn_rows:
        running += (t.deposit or 0.0) - (t.withdrawal or 0.0)
        if t.balance is not None:
            if abs(t.balance - running) > 0.01:
                raise ValueError(
                    f"Balance mismatch on {t.date} {t.description}: "
                    f"running={running:.2f} printed={t.balance:.2f}"
                )
            last_printed = t.balance
    if abs(last_printed - running) > 0.01:
        raise ValueError(f"Closing balance mismatch: running={running:.2f} printed={last_printed:.2f}")


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------

def extract_statement(pdf_path: str | Path, csv_out: str | Path | None = None) -> Path:
    pdf_path = Path(pdf_path)
    csv_out = Path(csv_out) if csv_out else pdf_path.with_suffix(".csv")

    all_rows: list[Row] = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            all_rows.extend(_extract_page_rows(page))
    finally:
        doc.close()

    transactions = _classify_rows(all_rows)
    _validate(transactions)

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for t in transactions:
            writer.writerow({
                "date": t.date,
                "description": t.description,
                "reference": t.reference,
                "type": t.type,
                "deposit": f"{t.deposit:.2f}" if t.deposit is not None else "",
                "withdrawal": f"{t.withdrawal:.2f}" if t.withdrawal is not None else "",
                "balance": f"{t.balance:.2f}" if t.balance is not None else "",
                "note": t.note,
            })
    return csv_out


if __name__ == "__main__":
    import sys
    out = extract_statement(sys.argv[1])
    print(f"wrote {out}")
