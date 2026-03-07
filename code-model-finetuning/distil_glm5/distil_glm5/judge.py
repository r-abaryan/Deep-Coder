"""Optional judge model: accepts or rejects teacher outputs (rejection sampling).

When enabled, uses a second model (e.g. Qwen-3.5 70B) to evaluate teacher outputs.
Judge prompts are task-type-aware so the reviewer evaluates code correctness,
style, and edge-case handling rather than just a blanket YES/NO.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from distil_glm5.teacher_client import OpenAICompatChatClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task-aware judge prompts
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a senior code reviewer and quality checker. "
    "Evaluate the assistant's response against the instruction. "
    "First, write a brief evaluation (2-4 sentences) covering correctness, "
    "edge-case handling, and code quality. "
    "Then give your final verdict on a new line: exactly YES (keep) or NO (reject)."
)

JUDGE_TEMPLATE_CODE = """Instruction:
{instruction}

Language: {language}

Assistant response:
{output}

Evaluate this code response:
1. Does the code appear syntactically correct and runnable?
2. Does it handle edge cases (empty inputs, nulls, large inputs)?
3. Is it idiomatic {language} with reasonable naming and structure?
4. Does it fully address what the instruction asked for?

Write a brief evaluation, then answer YES or NO on the final line."""

JUDGE_TEMPLATE_BUGFIX = """Instruction:
{instruction}

Language: {language}

Assistant response:
{output}

Evaluate this bugfix response:
1. Is the bug clearly identified and explained?
2. Does the fix actually resolve the issue without introducing new bugs?
3. Are edge cases that triggered the bug addressed?
4. Is the explanation accurate?

Write a brief evaluation, then answer YES or NO on the final line."""

JUDGE_TEMPLATE_REFACTOR = """Instruction:
{instruction}

Language: {language}

Assistant response:
{output}

Evaluate this refactoring response:
1. Is the refactored code meaningfully better than the original?
2. Does it apply clean code principles (naming, structure, error handling)?
3. Does it preserve the original functionality?
4. Is the explanation of changes clear?

Write a brief evaluation, then answer YES or NO on the final line."""

JUDGE_TEMPLATE_EXPLAIN = """Instruction:
{instruction}

Language: {language}

Assistant response:
{output}

Evaluate this explanation:
1. Is the technical content accurate?
2. Are code examples correct and illustrative?
3. Does it cover performance and security implications as relevant?
4. Is it complete enough to be useful?

Write a brief evaluation, then answer YES or NO on the final line."""

JUDGE_TEMPLATE_TESTS = """Instruction:
{instruction}

Language: {language}

Assistant response:
{output}

Evaluate these tests:
1. Do the tests cover both positive and negative cases?
2. Are assertion patterns correct for the testing framework used?
3. Do they test meaningful edge cases?
4. Are test names descriptive?

Write a brief evaluation, then answer YES or NO on the final line."""

JUDGE_TEMPLATE_GENERIC = """Instruction:
{instruction}

Language: {language}

Assistant response:
{output}

Does this response correctly and fully address the instruction?
Is the code (if any) correct, idiomatic, and well-structured?

Write a brief evaluation, then answer YES or NO on the final line."""

_TASK_TEMPLATES: dict[str, str] = {
    "completion": JUDGE_TEMPLATE_CODE,
    "bugfix": JUDGE_TEMPLATE_BUGFIX,
    "refactor": JUDGE_TEMPLATE_REFACTOR,
    "explain": JUDGE_TEMPLATE_EXPLAIN,
    "code_review": JUDGE_TEMPLATE_CODE,
    "tests": JUDGE_TEMPLATE_TESTS,
}


def get_instruction_from_row(row: dict[str, Any]) -> str:
    """Extract the user instruction from a raw or curated row (messages)."""
    messages = row.get("messages") or []
    for m in reversed(messages):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def _get_language_from_row(row: dict[str, Any]) -> str:
    """Extract language from a row."""
    return (row.get("language") or "unknown").strip()


def _parse_verdict(text: str) -> tuple[bool, str]:
    """
    Parse the judge response into (keep, reasoning).
    Looks for YES or NO on the last non-empty line.
    Returns the reasoning text and the boolean verdict.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return False, ""

    reasoning = "\n".join(lines[:-1]) if len(lines) > 1 else ""
    last_line = lines[-1].upper()

    if re.search(r"\bYES\b", last_line):
        return True, reasoning
    if re.search(r"\bNO\b", last_line):
        return False, reasoning

    # Fallback: scan the entire response
    if re.search(r"\bYES\b", text.upper()):
        return True, text
    return False, text


def judge_keep(
    *,
    instruction: str,
    output: str,
    judge_config: Any,
    client: OpenAICompatChatClient | None = None,
    task_type: str = "unknown",
    language: str = "unknown",
) -> tuple[bool, str]:
    """
    Ask the judge model whether to keep this (instruction, output) pair.

    Returns:
        (keep: bool, reasoning: str) — keep=True means accept, reasoning
        is the judge's evaluation text (logged for debugging).
    """
    if not instruction and not output:
        return False, "empty_input"

    template = _TASK_TEMPLATES.get(task_type, JUDGE_TEMPLATE_GENERIC)
    user_content = template.format(
        instruction=instruction,
        output=output,
        language=language,
    )

    payload = {
        "model": judge_config.model_id,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_content},
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

    try:
        resp = client.chat_completions(payload)
        text = resp.content or ""
        keep, reasoning = _parse_verdict(text)
        return keep, reasoning
    except Exception as e:
        logger.warning("Judge call failed: %s — defaulting to reject", e)
        return False, f"judge_error: {e}"