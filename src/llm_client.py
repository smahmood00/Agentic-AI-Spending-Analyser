"""OpenRouter wrapper. One entrypoint, role-based model routing, JSON-schema enforced."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LLMResponse:
    content: str
    parsed: Any
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


class LLMClient:
    def __init__(self, config_path: str | Path | None = None):
        config_path = Path(config_path) if config_path else ROOT / "config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.base_url = cfg["openrouter"]["base_url"]
        self.timeout = cfg["openrouter"]["timeout_seconds"]
        self.max_retries = cfg["openrouter"]["max_retries"]
        self.initial_backoff = cfg["openrouter"]["initial_backoff_seconds"]
        self.models = cfg["models"]
        self.log_path = ROOT / cfg["paths"]["llm_call_log"]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set in environment / .env")

    def _post(self, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        backoff = self.initial_backoff
        for attempt in range(self.max_retries):
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if r.status_code == 429 or r.status_code >= 500:
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"LLM request failed after {self.max_retries} retries")

    def complete(
        self,
        role: str,
        messages: list[dict],
        schema: dict | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if role not in self.models:
            raise ValueError(f"Unknown role '{role}'. Configured: {list(self.models)}")
        model = self.models[role]

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": f"{role}_output",
                    "strict": True,
                    "schema": schema,
                },
            }

        t0 = time.time()
        body = self._post(payload)
        latency_ms = int((time.time() - t0) * 1000)

        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        parsed: Any = None
        if schema is not None:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(f"LLM returned non-JSON despite schema: {content[:300]}") from e

        self._log_call(role, model, input_tokens, output_tokens, latency_ms)

        return LLMResponse(
            content=content,
            parsed=parsed,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

    def _log_call(self, role: str, model: str, in_tok: int, out_tok: int, latency_ms: int) -> None:
        entry = {
            "ts": time.time(),
            "role": role,
            "model": model,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_ms": latency_ms,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
