from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class ChatResponse:
    content: str
    raw: dict[str, Any]


class OpenAICompatChatClient:
    """
    Minimal client for OpenAI-compatible chat completions endpoints (e.g., vLLM).

    This is intentionally small and dependency-light. If you prefer the official OpenAI SDK,
    we can swap later.
    """

    def __init__(self, *, base_url: str, api_key: str, timeout_s: float = 120, max_retries: int = 4):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def chat_completions(self, payload: dict[str, Any]) -> ChatResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout_s)
                resp.raise_for_status()
                data = resp.json()
                # OpenAI-style: choices[0].message.content
                content = data["choices"][0]["message"].get("content", "") or ""
                return ChatResponse(content=content, raw=data)
            except Exception as e:  # noqa: BLE001 - boundary layer
                last_err = e
                if attempt >= self.max_retries:
                    break
                # simple backoff
                time.sleep(min(2.0**attempt, 10.0))
        raise RuntimeError(f"Chat request failed after retries: {last_err}")


def build_chat_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    n: int = 1,
    stop: list[str] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": n,
    }
    if stop:
        payload["stop"] = stop
    if seed is not None:
        # OpenAI supports seed on some APIs; vLLM may ignore it depending on version.
        payload["seed"] = seed
    return payload

