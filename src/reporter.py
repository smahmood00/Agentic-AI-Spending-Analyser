"""Reporter agent: one batched LLM call. Input = metrics.json only."""
from __future__ import annotations

import json

from src.llm_client import LLMClient

SYSTEM_PROMPT = """You are a personal finance analyst writing a spending report.

You will be given a JSON object of pre-computed metrics. Your job is to render it as a
clean Markdown report with EXACTLY these 7 sections, in order:

1. ## Summary
2. ## Category Breakdown
3. ## Top Merchants
4. ## Anomalies
5. ## Recurring Expenses
6. ## Needs Your Review
7. ## Observations

Rules — these are hard constraints, not suggestions:
- Use ONLY numbers and merchant names that appear in the metrics JSON. Never invent
  figures, never round in a way that produces a number not in the JSON.
- NEVER guess the purpose of a P2P transfer or any transaction in "needs_review".
  Refer to them as "Unknown Purpose" — do not call them rent, salary, family, loan, etc.
- Do not claim a merchant is "recurring" unless it appears in the recurring list.
- "Observations" should describe patterns visible in the metrics. No advice or
  prescriptive language ("you should...", "consider..."). Just neutral pattern callouts.
- Format currency as `HKD <amount>` (no symbol guessing).
- Be concise. The full report should fit on one screen if possible.
"""


def generate_report(client: LLMClient, metrics: dict, critic_feedback: list[str] | None = None) -> str:
    user_content = "Metrics:\n```json\n" + json.dumps(metrics, indent=2, default=str) + "\n```"
    if critic_feedback:
        user_content += (
            "\n\nA prior draft was rejected by the critic for these issues. "
            "Address each one in this revision:\n"
            + "\n".join(f"- {issue}" for issue in critic_feedback)
        )
    resp = client.complete(
        role="reporter",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.content
