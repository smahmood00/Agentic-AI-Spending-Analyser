# AI Financial Analyst

An agentic application that processes HSBC and Hang Seng bank statements (PDF or CSV) and produces a structured spending report. Built on a **LangGraph** multi-agent pipeline with human-in-the-loop categorisation, LLM-based reporting, and a critic reflection pass.

---

## Architecture

```
PDF / CSV upload
      │
      ▼
┌─────────────┐
│    parse    │  Deterministic — pandas CSV parser
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  categorize │  Cache → rules → batched LLM (one call regardless of row count)
└──────┬──────┘
       │ hitl_pending?
      yes ──────────────────────────────────┐
       │ no                                 ▼
       │                         ┌─────────────────────┐
       │                         │    human_review     │  interrupt() ← user labels
       │                         └──────────┬──────────┘
       │◄──────────────────────────────────┘
       ▼
┌─────────────┐
│   analyze   │  Deterministic — pandas aggregations → metrics.json
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   report    │  LLM drafts 7-section Markdown report from metrics only
└──────┬──────┘
       │
       ▼
┌─────────────┐    rejected & revision_count ≤ 1
│   critique  │ ─────────────────────────────────► report (revision pass)
└──────┬──────┘
       │ approved (or max revisions reached)
       ▼
      END
```

### Agentic patterns used

| Pattern | Where |
|---|---|
| **Multi-agent** | Six specialised nodes coordinated by LangGraph StateGraph |
| **Human-in-the-loop** | `interrupt()` pauses the graph; `Command(resume=)` continues it after user labels |
| **Reflection** | Critic LLM reviews the Reporter's draft against `metrics.json`; rejects grounding violations |
| **Conditional edges** | Skip HITL when cache/rules cover all transactions; limit critic to one revision cycle |
| **State persistence** | `MemorySaver` checkpoints state between HITL pause and resume (pickle serde for DataFrame compatibility) |

---

## Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — graph topology, state management, HITL interrupt
- **[Streamlit](https://streamlit.io)** — web UI (upload, HITL form, report viewer, PDF download)
- **[PyMuPDF](https://pymupdf.readthedocs.io)** — PDF text extraction and PDF report rendering
- **[OpenRouter](https://openrouter.ai)** — routes to `claude-haiku-4.5` (categoriser) and `claude-sonnet-4.5` (reporter, critic)
- **pandas** — all numeric computation (the LLM never does math)

---

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY
```

---

## Usage

### Streamlit UI (recommended)

```bash
streamlit run app.py
```

1. Upload a PDF or CSV bank statement (HSBC or Hang Seng).
2. If any transactions need labelling, a form appears — pick from the category list or type a custom one.
3. The pipeline resumes automatically and displays the report with download buttons (Markdown + PDF).

### CLI

```bash
python src/main.py data/statement.csv
```

The pipeline streams node-level progress to stdout and prompts for labels via `stdin` when HITL fires.

---

## Project structure

```
ai-financial-analyst/
├── app.py                          Streamlit UI (upload → HITL → report)
├── config.yaml                     Models, thresholds, paths
├── requirements.txt
├── .env.example
├── src/
│   ├── pdf_extractor.py            PDF → CSV (HSBC + Hang Seng, PyMuPDF word-bbox)
│   ├── parser.py                   CSV → typed DataFrame
│   ├── categorizer.py              3-tier: cache → rules → batched LLM → HITL
│   ├── analyst.py                  Pandas aggregations → metrics.json
│   ├── reporter.py                 LLM → 7-section Markdown report
│   ├── critic.py                   LLM reflection / grounding check
│   ├── llm_client.py               OpenRouter wrapper (retry, logging, JSON schema)
│   └── graph/
│       ├── state.py                PipelineState TypedDict
│       ├── nodes.py                One function per graph node
│       ├── edges.py                Conditional routing functions
│       └── graph.py                StateGraph assembly + _PickleSerde
├── tests/
│   ├── eval_categorizer.py         Offline + live categoriser evaluation
│   └── fixtures/categorizer_gold.json
└── data/                           (gitignored — put statements here)
```

---

## The agents

**Parser** — loads CSV, infers year for `DD Mon` dates (handles Dec→Jan rollover), computes signed `amount`, forward-fills balance.

**Categoriser** — three tiers, LLM only as last resort:
1. Cache lookup (`data/merchant_labels.json`) — warm runs skip LLM entirely.
2. Rule-based regex for known patterns (`PAYME`, `ATM WITHDRAWAL`, `CASH REBATE`, card payments).
3. One batched LLM call for all remaining unknowns — one call regardless of row count.

HITL fires for P2P transfers to person names and for any LLM label below the confidence threshold. Labels are written back to cache so subsequent runs are fully automatic.

**Analyst** — computes category totals, top merchants, cashflow summary, IQR-based anomaly detection, and recurring expense detection (≥3 occurrences across distinct months). Writes `metrics.json`.

**Reporter** — receives only `metrics.json` (no raw transactions). Produces a 7-section Markdown report. The prompt explicitly forbids invented numbers, invented merchants, and purpose inference for P2P items.

**Critic** — reflects on the draft against `metrics.json`. Rejects: unsupported numbers, P2P purpose inference, false recurrence claims, prescriptive financial advice. The graph allows at most one revision pass.

**PDF Extractor** — uses PyMuPDF word bboxes and visual row grouping (y-coordinate rounding) to reconstruct the tabular structure from raw PDF text. Classifies words into columns by x-position boundaries derived from the header row. Handles both HSBC and Hang Seng statement formats using bank-specific reference code regexes and footer markers.

---

## Configuration

`config.yaml` is the only place models, thresholds, and paths are set — no code changes needed to swap a model:

```yaml
models:
  categorizer: anthropic/claude-haiku-4.5   # cheap — one batched call
  reporter:    anthropic/claude-sonnet-4.5
  critic:      anthropic/claude-sonnet-4.5

thresholds:
  llm_confidence_min:        0.75   # below this → HITL
  recurring_min_occurrences: 3      # ≥3 distinct months = recurring
  anomaly_iqr_multiplier:    1.5    # IQR outlier threshold
```

---

## LLM cost profile

| Run type | LLM calls |
|---|---|
| Cold (empty cache) | 3 — categoriser (1 batched) + reporter + critic |
| Warm (cache populated) | 2 — reporter + critic only |
| Critic rejects once | +1 reporter call |

The categoriser prompt scales with the number of *unknown* rows, not total rows. The reporter receives only `metrics.json`, so its prompt is bounded by category count, not transaction count.

---

## Evaluation

```bash
python tests/eval_categorizer.py --offline   # stub LLM, no API spend
python tests/eval_categorizer.py             # live OpenRouter call
```

Enforced metrics:
- `rule_precision ≥ 0.98`
- `overall_accuracy ≥ 0.90`
- `hitl_recall_p2p_person == 1.0` — every P2P-to-person row must reach HITL on cold run
- `llm_calls_cold == 1` — batching contract
- `llm_calls_warm == 0` — cache contract

---

## Design decisions

**Why LangGraph over plain Python?** The HITL requirement is the key driver. LangGraph's `interrupt()` / `Command(resume=)` cleanly serialises state between the pause and resume without any session-state patching. The StateGraph also makes the conditional routing (skip HITL, limit revisions) explicit and testable.

**Why a separate Critic agent?** The Reporter is given only metrics and told to narrate — it is optimistic. The Critic is adversarial and grounded against the same metrics, catching fabricated numbers or unsupported claims before they reach the user.

**Why does the LLM never see raw transactions?** Privacy and prompt-size control. The Analyst reduces N transactions to a bounded `metrics.json`. The Reporter and Critic only ever see that summary.

**Why pickle for MemorySaver?** LangGraph's default msgpack serde cannot serialise pandas DataFrames or custom dataclasses. Pickle supports arbitrary Python objects and is safe here because the checkpointer is in-memory only — state never leaves the process.
