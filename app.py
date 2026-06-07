"""Streamlit UI for the AI Financial Analyst pipeline.

Stages (tracked in st.session_state.stage):
  upload  → user uploads a PDF or CSV bank statement
  hitl    → user labels low-confidence / P2P transactions
  running → analyze → report → critique LLM calls execute
  done    → report and metrics are displayed

The LangGraph pipeline handles all state transitions and HITL via its
interrupt / Command(resume=...) mechanism. Streamlit session_state is only
used to persist the compiled graph and thread config across rerenders.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path

import fitz
import markdown as md_lib
import streamlit as st
import yaml
from langgraph.types import Command

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.categorizer import CATEGORIES
from src.graph import build_pipeline
from src.graph.state import PipelineState
from src.llm_client import LLMClient
from src.pdf_extractor import extract_statement


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _report_to_pdf_bytes(report_md: str) -> bytes:
    """Render a Markdown report to PDF bytes using PyMuPDF Story."""
    html = (
        "<html><body style='font-family:Helvetica;font-size:11pt;line-height:1.6'>"
        + md_lib.markdown(report_md, extensions=["tables"])
        + "</body></html>"
    )
    buf = io.BytesIO()
    writer = fitz.DocumentWriter(buf)
    story = fitz.Story(html)
    mediabox = fitz.paper_rect("a4")
    margin = 50
    where = mediabox + (margin, margin, -margin, -margin)
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    return buf.getvalue()


def _build_thread_config(app_cfg: dict) -> dict:
    """Create a per-session LangGraph thread config."""
    return {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "llm_client": LLMClient(config_path=str(ROOT / "config.yaml")),
            "app_config": app_cfg,
        }
    }


# ── Stage: upload ─────────────────────────────────────────────────────────────

def render_upload() -> None:
    st.subheader("Upload Bank Statement")
    uploaded = st.file_uploader(
        "(PDF or CSV)",
        type=["pdf", "csv"],
    )
    if uploaded is None:
        return

    if not st.button("Analyse", type="primary"):
        return

    app_cfg = _load_config()

    with st.status("Preparing...", expanded=True) as status:
        # Extract PDF to CSV if necessary.
        suffix = ".pdf" if uploaded.name.lower().endswith(".pdf") else ".csv"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        if suffix == ".pdf":
            st.write("Extracting PDF to CSV...")
            csv_path = extract_statement(tmp_path, tmp_path.with_suffix(".csv"))
        else:
            csv_path = tmp_path

        # Build the pipeline once per session so MemorySaver persists across rerenders.
        pipeline = build_pipeline()
        thread_config = _build_thread_config(app_cfg)

        st.write("Running categorisation (LLM call on cold run)...")
        initial_state = PipelineState(
            csv_path=str(csv_path),
            parsed=None,
            cat_result=None,
            hitl_pending=[],
            metrics=None,
            report=None,
            verdict=None,
            revision_count=0,
        )
        pipeline.invoke(initial_state, config=thread_config)
        status.update(label="Ready", state="complete")

    # Persist pipeline and config so HITL and running stages can continue it.
    st.session_state.pipeline = pipeline
    st.session_state.thread_config = thread_config
    st.session_state.app_cfg = app_cfg

    snapshot = pipeline.get_state(thread_config)
    if snapshot.next:
        # Pipeline paused at human_review — expose pending transactions.
        st.session_state.hitl_pending = snapshot.tasks[0].interrupts[0].value
        st.session_state.stage = "hitl"
    else:
        # No HITL needed — pipeline already ran to completion.
        st.session_state.result = snapshot.values
        st.session_state.stage = "done"

    st.rerun()


# ── Stage: hitl ───────────────────────────────────────────────────────────────

def render_hitl() -> None:
    # Second pass: form already submitted — run the pipeline with a full-screen spinner.
    # The form is not rendered so the button cannot be clicked again.
    if st.session_state.get("hitl_processing"):
        with st.spinner("Running analysis — this may take up to 30 seconds..."):
            st.session_state.pipeline.invoke(
                Command(resume=st.session_state.pop("hitl_user_labels")),
                config=st.session_state.thread_config,
            )
        st.session_state.result = st.session_state.pipeline.get_state(
            st.session_state.thread_config
        ).values
        del st.session_state["hitl_processing"]
        st.session_state.stage = "done"
        st.rerun()
        return

    # First pass: show the labelling form.
    pending: list[dict] = st.session_state.hitl_pending
    st.subheader(f"Review Transactions — {len(pending)} item(s) need your input")
    st.caption(
        "These transactions could not be categorised automatically. "
        "Pick from the list or type a custom category."
    )

    with st.form("hitl_form"):
        user_labels: dict[str, str] = {}

        for i, txn in enumerate(pending):
            st.divider()
            col_info, col_amount = st.columns([3, 1])
            with col_info:
                st.markdown(f"**{txn['description']}**")
                st.caption(f"{txn['date_str']}  ·  reason: {txn['reason']}")
            with col_amount:
                colour = "green" if txn["amount"] >= 0 else "red"
                st.markdown(
                    f"<span style='color:{colour};font-size:1.1rem'>"
                    f"HKD {txn['amount']:+,.2f}</span>",
                    unsafe_allow_html=True,
                )

            options = CATEGORIES + ["Custom..."]
            choice = st.selectbox("Category", options, key=f"cat_{i}")
            if choice == "Custom...":
                custom = st.text_input(
                    "Enter custom category",
                    key=f"custom_{i}",
                    placeholder="e.g. Rent, School fees, Gym",
                )
                user_labels[txn["description"]] = custom.strip() or "Other"
            else:
                user_labels[txn["description"]] = choice

        st.divider()
        submitted = st.form_submit_button("Submit & Continue", type="primary")

    if submitted:
        # Save labels and rerun — the second pass will show the spinner and run the pipeline.
        st.session_state.hitl_user_labels = user_labels
        st.session_state.hitl_processing = True
        st.rerun()


# ── Stage: running ────────────────────────────────────────────────────────────

def render_running() -> None:
    if "result" in st.session_state:
        st.session_state.stage = "done"
        st.rerun()
        return

    pipeline = st.session_state.pipeline
    thread_config = st.session_state.thread_config

    with st.status("Running analysis...", expanded=True) as status:
        st.write("[3/5] Computing metrics...")
        st.write("[4/5] Generating report draft (LLM)...")
        st.write("[5/5] Critic review (LLM)...")

        # The graph resumes from analyze onward (categorise is already done).
        # We pass None as input because the graph resumes from its checkpointed state.
        pipeline.invoke(None, config=thread_config)
        status.update(label="Analysis complete!", state="complete")

    st.session_state.result = pipeline.get_state(thread_config).values
    st.session_state.stage = "done"
    st.rerun()


# ── Stage: done ───────────────────────────────────────────────────────────────

def render_done() -> None:
    result = st.session_state.result
    app_cfg = st.session_state.app_cfg
    report = result["report"]
    metrics = result["metrics"]

    # Summary metrics row.
    cf = metrics.get("cashflow", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income",   f"HKD {cf.get('total_deposits', 0):,.2f}")
    col2.metric("Total Spending", f"HKD {cf.get('total_withdrawals', 0):,.2f}")
    col3.metric("Savings Rate",   f"{cf.get('savings_rate', 0) * 100:.1f}%")

    st.divider()
    st.markdown(report)

    # Download buttons.
    col_md, col_pdf, _ = st.columns([1, 1, 4])
    with col_md:
        st.download_button(
            "Download Markdown",
            data=report,
            file_name="spending_report.md",
            mime="text/markdown",
        )
    with col_pdf:
        if "report_pdf" not in st.session_state:
            st.session_state.report_pdf = _report_to_pdf_bytes(report)
        st.download_button(
            "Download PDF",
            data=st.session_state.report_pdf,
            file_name="spending_report.pdf",
            mime="application/pdf",
        )

    # LLM call summary.
    log_path = ROOT / app_cfg["paths"]["llm_call_log"]
    if log_path.exists():
        counts: Counter = Counter()
        in_tok: Counter = Counter()
        out_tok: Counter = Counter()
        with open(log_path) as f:
            for line in f:
                e = json.loads(line)
                counts[e["role"]] += 1
                in_tok[e["role"]] += e["input_tokens"]
                out_tok[e["role"]] += e["output_tokens"]
        with st.expander("LLM call summary (cumulative across all runs)"):
            for role in sorted(counts):
                st.text(
                    f"{role:12s}  calls={counts[role]:3d}  "
                    f"in={in_tok[role]:6d} tokens  out={out_tok[role]:6d} tokens"
                )

    st.divider()
    if st.button("Analyse another statement"):
        for key in ["stage", "pipeline", "thread_config", "app_cfg",
                    "hitl_pending", "result", "report_pdf"]:
            st.session_state.pop(key, None)
        st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Financial Analyst",
    page_icon=":bar_chart:",
    layout="wide",
)
st.title("AI Financial Analyst")

if "stage" not in st.session_state:
    st.session_state.stage = "upload"

{
    "upload":  render_upload,
    "hitl":    render_hitl,
    "running": render_running,
    "done":    render_done,
}[st.session_state.stage]()
