from __future__ import annotations

import random
import uuid
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PromptExample:
    id: str
    task_type: str
    language: str
    messages: list[dict[str, str]]
    meta: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        # keep schema stable
        d["schema_version"] = 1
        return d


SYSTEM_CODE = "You are an expert software engineer. Be correct, concise, and practical."


def _new_id() -> str:
    return uuid.uuid4().hex


def build_prompt_pool(
    *,
    total: int,
    task_mix: dict[str, float],
    languages: list[str],
    seed: int | None = 42,
) -> list[PromptExample]:
    """
    Minimal prompt pool builder.

    This intentionally starts simple: we generate templated prompts for a few task types.
    Later we will extend with OSS-grounded tasks and Evol-Instruct mutations.
    """
    rng = random.Random(seed)
    languages = languages or ["python"]

    # normalize task mix
    if not task_mix:
        task_mix = {"completion": 1.0}
    total_weight = sum(float(v) for v in task_mix.values())
    probs = [(k, float(v) / total_weight) for k, v in task_mix.items()]

    def sample_task() -> str:
        x = rng.random()
        acc = 0.0
        for name, p in probs:
            acc += p
            if x <= acc:
                return name
        return probs[-1][0]

    out: list[PromptExample] = []
    for _ in range(total):
        task = sample_task()
        lang = rng.choice(languages)
        ex = _make_templated_prompt(rng=rng, task_type=task, language=lang)
        out.append(ex)
    return out


def _make_templated_prompt(*, rng: random.Random, task_type: str, language: str) -> PromptExample:
    # Language-aware templates. Defaults to Python if unknown.
    norm_lang = language.lower()

    if norm_lang in {"python", "py"}:
        language = "python"
        if task_type == "completion":
            user = (
                "Complete the following Python code.\n\n"
                "Requirements:\n"
                "- Keep it idiomatic and type-hinted\n"
                "- Handle edge cases\n\n"
                "Starter code:\n"
                "```python\n"
                "def normalize_whitespace(text: str) -> str:\n"
                "    \"\"\"Collapse multiple spaces and trim.\"\"\"\n"
                "    # TODO\n"
                "```\n"
            )
        elif task_type == "bugfix":
            user = (
                "Fix the bug(s) in this Python code and return only the corrected code.\n\n"
                "```python\n"
                "def is_palindrome(s: str) -> bool:\n"
                "    s = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
                "    return s == s.reverse()\n"
                "```\n"
            )
        elif task_type == "refactor":
            user = (
                "Refactor this Python code for readability and performance without changing behavior.\n"
                "Return only the refactored code.\n\n"
                "```python\n"
                "def count_words(text):\n"
                "    words = text.split(' ')\n"
                "    d = {}\n"
                "    for w in words:\n"
                "        if w in d:\n"
                "            d[w] = d[w] + 1\n"
                "        else:\n"
                "            d[w] = 1\n"
                "    return d\n"
                "```\n"
            )
        elif task_type == "tests":
            user = (
                "Write unit tests (pytest) for this function. Include edge cases.\n\n"
                "```python\n"
                "def clamp(x: float, lo: float, hi: float) -> float:\n"
                "    return max(lo, min(hi, x))\n"
                "```\n"
            )
        elif task_type == "explain":
            user = (
                "Explain what this Python function does, its time complexity, and common pitfalls.\n\n"
                "```python\n"
                "def dedupe_keep_order(items):\n"
                "    seen = set()\n"
                "    out = []\n"
                "    for x in items:\n"
                "        if x in seen:\n"
                "            continue\n"
                "        seen.add(x)\n"
                "        out.append(x)\n"
                "    return out\n"
                "```\n"
            )
        elif task_type == "code_review":
            user = (
                "Do a short code review: list issues and suggest improvements.\n\n"
                "```python\n"
                "import requests\n"
                "def fetch(url):\n"
                "    return requests.get(url).text\n"
                "```\n"
            )
        else:
            user = (
                "Write a Python function that parses an ISO date string and returns a datetime.date."
            )
            task_type = "misc"

    elif norm_lang in {"typescript", "ts"}:
        language = "typescript"
        if task_type == "completion":
            user = (
                "Complete the following TypeScript function.\n\n"
                "Requirements:\n"
                "- Use precise types\n"
                "- Do not use `any`\n"
                "- Handle edge cases\n\n"
                "Starter code:\n"
                "```ts\n"
                "export function normalizeWhitespace(text: string): string {\n"
                "  // TODO\n"
                "}\n"
                "```\n"
            )
        elif task_type == "bugfix":
            user = (
                "Fix the bug(s) in this TypeScript code and return only the corrected code.\n\n"
                "```ts\n"
                "export function isPalindrome(s: string): boolean {\n"
                "  const cleaned = s.toLowerCase().replace(/[^a-z0-9]/g, '');\n"
                "  return cleaned === cleaned.reverse();\n"
                "}\n"
                "```\n"
            )
        elif task_type == "refactor":
            user = (
                "Refactor this TypeScript code for readability and performance without changing behavior.\n"
                "Return only the refactored code.\n\n"
                "```ts\n"
                "export function countWords(text: string): Record<string, number> {\n"
                "  const parts = text.split(' ');\n"
                "  const counts: Record<string, number> = {};\n"
                "  for (const w of parts) {\n"
                "    counts[w] = (counts[w] ?? 0) + 1;\n"
                "  }\n"
                "  return counts;\n"
                "}\n"
                "```\n"
            )
        elif task_type == "tests":
            user = (
                "Write unit tests (Vitest or Jest) for this TypeScript function. Include edge cases.\n\n"
                "```ts\n"
                "export function clamp(x: number, lo: number, hi: number): number {\n"
                "  return Math.max(lo, Math.min(hi, x));\n"
                "}\n"
                "```\n"
            )
        elif task_type == "explain":
            user = (
                "Explain what this TypeScript function does, its time complexity, and common pitfalls.\n\n"
                "```ts\n"
                "export function dedupeKeepOrder<T>(items: T[]): T[] {\n"
                "  const seen = new Set<T>();\n"
                "  const out: T[] = [];\n"
                "  for (const x of items) {\n"
                "    if (seen.has(x)) continue;\n"
                "    seen.add(x);\n"
                "    out.push(x);\n"
                "  }\n"
                "  return out;\n"
                "}\n"
                "```\n"
            )
        elif task_type == "code_review":
            user = (
                "Do a short code review: list issues and suggest improvements to typing and error handling.\n\n"
                "```ts\n"
                "import axios from 'axios';\n"
                "export async function fetch(url: string) {\n"
                "  const res = await axios.get(url);\n"
                "  return res.data;\n"
                "}\n"
                "```\n"
            )
        else:
            user = "Write a small TypeScript utility function for parsing ISO date strings."
            task_type = "misc"

    elif norm_lang in {"javascript", "js"}:
        language = "javascript"
        if task_type == "completion":
            user = (
                "Complete the following JavaScript function.\n\n"
                "Requirements:\n"
                "- Use modern ES syntax\n"
                "- Handle edge cases\n\n"
                "Starter code:\n"
                "```js\n"
                "export function normalizeWhitespace(text) {\n"
                "  // TODO\n"
                "}\n"
                "```\n"
            )
        elif task_type == "bugfix":
            user = (
                "Fix the bug(s) in this JavaScript code and return only the corrected code.\n\n"
                "```js\n"
                "export function isPalindrome(s) {\n"
                "  const cleaned = s.toLowerCase().replace(/[^a-z0-9]/g, '');\n"
                "  return cleaned === cleaned.reverse();\n"
                "}\n"
                "```\n"
            )
        elif task_type == "refactor":
            user = (
                "Refactor this JavaScript code for readability and performance without changing behavior.\n"
                "Return only the refactored code.\n\n"
                "```js\n"
                "export function countWords(text) {\n"
                "  const parts = text.split(' ');\n"
                "  const counts = {};\n"
                "  for (const w of parts) {\n"
                "    counts[w] = (counts[w] ?? 0) + 1;\n"
                "  }\n"
                "  return counts;\n"
                "}\n"
                "```\n"
            )
        elif task_type == "tests":
            user = (
                "Write Jest tests for this JavaScript function. Include edge cases.\n\n"
                "```js\n"
                "export function clamp(x, lo, hi) {\n"
                "  return Math.max(lo, Math.min(hi, x));\n"
                "}\n"
                "```\n"
            )
        elif task_type == "explain":
            user = (
                "Explain what this JavaScript function does, its time complexity, and common pitfalls.\n\n"
                "```js\n"
                "export function dedupeKeepOrder(items) {\n"
                "  const seen = new Set();\n"
                "  const out = [];\n"
                "  for (const x of items) {\n"
                "    if (seen.has(x)) continue;\n"
                "    seen.add(x);\n"
                "    out.push(x);\n"
                "  }\n"
                "  return out;\n"
                "}\n"
                "```\n"
            )
        elif task_type == "code_review":
            user = (
                "Do a short code review: list issues and suggest improvements (error handling, async, style).\n\n"
                "```js\n"
                "import axios from 'axios';\n"
                "export async function fetch(url) {\n"
                "  return (await axios.get(url)).data;\n"
                "}\n"
                "```\n"
            )
        else:
            user = "Write a small JavaScript utility function for parsing ISO date strings."
            task_type = "misc"

    else:
        # Fallback: treat as Python.
        language = "python"
        user = (
            "Write a Python function that parses an ISO date string and returns a datetime.date."
        )
        task_type = "misc"

    # Add a small amount of variation to reduce duplicates.
    style_hint = rng.choice(
        [
            "Prefer small helper functions.",
            "Prefer clear variable names.",
            "Prefer early returns.",
            "Avoid unnecessary dependencies.",
        ]
    )
    user = user + f"\nStyle hint: {style_hint}\n"

    return PromptExample(
        id=_new_id(),
        task_type=task_type,
        language=language,
        messages=[
            {"role": "system", "content": SYSTEM_CODE},
            {"role": "user", "content": user},
        ],
        meta={"source": "templated_v0"},
    )

