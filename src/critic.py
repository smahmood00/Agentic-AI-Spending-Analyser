"""Critic agent: reflection pass on the draft report against the metrics."""
from __future__ import annotations

import json

from src.llm_client import LLMClient

CRITIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["approved", "issues"],
    "properties": {
        "approved": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

SYSTEM_PROMPT = """You are a strict reviewer of a personal finance report.

You receive:
1. A draft Markdown report.
2. The source metrics JSON the report was generated from.

Reject the report (approved=false) if ANY of the following are true:
- A dollar figure or percentage appears in the report that is NOT in the metrics.
- A merchant name in the report is not in the metrics.
- The report infers the purpose of a P2P transfer or any item in "needs_review"
  (e.g., calls a transfer "rent", "salary", "family transfer", "loan").
- The report claims a merchant is recurring when it is not in the recurring list.
- The report contains prescriptive advice ("you should...", "consider doing X...").
- Any of the 7 required sections are missing: Summary, Category Breakdown, Top
  Merchants, Anomalies, Recurring Expenses, Needs Your Review, Observations.

If approved, return approved=true and issues=[]. If rejected, list each violation
concretely (one issue per problem, naming the offending text/number) so the writer
can fix it.
"""


def review(client: LLMClient, report_md: str, metrics: dict) -> dict:
    user_content = (
        "Draft report:\n```markdown\n"
        + report_md
        + "\n```\n\nSource metrics:\n```json\n"
        + json.dumps(metrics, indent=2, default=str)
        + "\n```"
    )
    resp = client.complete(
        role="critic",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        schema=CRITIC_SCHEMA,
    )
    return resp.parsed
