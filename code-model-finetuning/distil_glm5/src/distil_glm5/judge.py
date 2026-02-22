"""Optional judge model: accepts or rejects teacher outputs (rejection sampling)."""

from __future__ import annotations

import re
from typing import Any

from distil_glm5.teacher_client import OpenAICompatChatClient

JUDGE_SYSTEM = (
    "You are a quality checker. Answer only YES or NO."
)
JUDGE_USER_TEMPLATE = """Instruction:
{instruction}

Assistant response:
{output}

Does this response correctly and fully address the instruction? Answer with exactly one word: YES or NO."""


def get_instruction_from_row(row: dict[str, Any]) -> str:
    """Extract the user instruction from a raw or curated row (messages)."""
    messages = row.get("messages") or []
    for m in reversed(messages):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def judge_keep(
    *,
    instruction: str,
    output: str,
    judge_config: Any,
    client: OpenAICompatChatClient | None = None,
) -> bool:
    """
    Ask the judge model whether to keep this (instruction, output) pair.
    Returns True to keep, False to reject.
    If judge is disabled (caller should not call when disabled), we return True.
    """
    if not instruction and not output:
        return False

    payload = {
        "model": judge_config.model_id,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(instruction=instruction, output=output)},
        ],
        "max_tokens": judge_config.max_tokens,
        "temperature": judge_config.temperature,
    }

    if client is None:
        client = OpenAICompatChatClient(
            base_url=judge_config.base_url,
            api_key=judge_config.api_key,
            timeout_s=judge_config.timeout_s,
            max_retries=judge_config.max_retries,
        )

    resp = client.chat_completions(payload)
    text = (resp.content or "").strip().upper()
    if re.search(r"\bYES\b", text):
        return True
    return False
