from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptExample:
    id: str
    task_type: str
    language: str
    messages: list[dict[str, str]]
    meta: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = 1
        return d


# ---------------------------------------------------------------------------
# System prompts — now with chain-of-thought (CoT) variants
# ---------------------------------------------------------------------------

SYSTEM_CODE = "You are an expert software engineer. Be correct, concise, and practical."

SYSTEM_CODE_COT = (
    "You are an expert software engineer. "
    "First, think through your approach step by step inside <think>...</think> tags. "
    "Then provide your final code or answer outside those tags. "
    "Be correct, concise, and practical."
)

SYSTEM_VARIANTS = [
    SYSTEM_CODE,
    "You are a senior software developer with years of experience. Provide clear, correct, and robust code.",
    "You are an expert programmer. Output high-quality code and minimal fluff.",
    "You are an AI coding assistant. Ensure your solutions are optimal and thoroughly commented.",
]

SYSTEM_VARIANTS_COT = [
    SYSTEM_CODE_COT,
    (
        "You are a senior software developer. "
        "Before writing code, reason step by step inside <think>...</think> tags "
        "about the approach, edge cases, and trade-offs. Then provide the final solution."
    ),
    (
        "You are an expert programmer. "
        "Think through the problem carefully in <think>...</think> tags first, "
        "then output high-quality code."
    ),
    (
        "You are an AI coding assistant. "
        "Always show your reasoning process inside <think>...</think> tags before "
        "providing the optimised, well-commented solution."
    ),
]


def _new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Topic / constraint / format pools
# ---------------------------------------------------------------------------

TOPICS = [
    "string manipulation", "data validation", "API request handling",
    "database schema querying", "data serialization", "matrix operations",
    "regex parsing", "file reading and writing", "graph traversal",
    "binary tree manipulation", "sorting arrays", "hashing data",
    "encryption/decryption wrappers", "user authentication logic",
    "logging middleware", "background job processing", "HTML parsing",
    "CLI argument parsing", "image processing", "rate limiting",
    "websocket communication", "caching mechanisms",
    "retry logic with exponential backoff", "date and timezone manipulation",
    "concurrent task execution", "configuration management",
    "state management", "custom array iteration",
    "mathematical computations", "memory caching",
]

CONSTRAINTS = [
    "Make it highly optimized for memory usage.",
    "Do not use any third-party libraries; stick to the standard library.",
    "Use a functional programming style.",
    "Use an object-oriented approach with appropriate abstractions.",
    "Include exhaustive error handling and distinct custom exception types.",
    "Ensure the time complexity is minimal (e.g. O(1) or O(N) where possible).",
    "Include detailed docstrings and comments explaining the edge cases.",
    "Make it thread-safe or safe for concurrent use.",
    "Write as clean and readable code as possible.",
    "Handle very large inputs cleanly without crashing or out-of-memory errors.",
    "Use strict type annotations for all parameters and return types.",
    "Use idiomatic patterns standard for this language.",
]

FORMATS = [
    "a single function",
    "a class with accompanying methods",
    "a small utility module",
    "a set of purely functional helpers",
]

STYLE_HINTS = [
    "Prefer clear variable names.",
    "Write modular code.",
    "Use comprehensive type annotations where applicable.",
    "Make sure edge cases like null or empty inputs are handled.",
    "Minimize nesting by using early returns.",
    "Avoid magic numbers or hardcoded strings.",
]


# ---------------------------------------------------------------------------
# OSS code seeds — for bugfix / refactor / code_review tasks
# ---------------------------------------------------------------------------

def load_oss_seeds(seed_path: str | Path | None) -> list[dict[str, Any]]:
    """
    Load OSS code seeds from a JSONL file.

    Each line should be a JSON object with at least:
        {"language": "python", "code": "...", "source": "the-stack-v2", "license": "MIT"}

    Returns an empty list if path is None or file doesn't exist.
    """
    if seed_path is None:
        return []
    p = Path(seed_path)
    if not p.exists():
        logger.warning("OSS seed file %s not found — falling back to templated-only prompts.", p)
        return []
    seeds: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "code" in obj and "language" in obj:
                    seeds.append(obj)
            except json.JSONDecodeError:
                continue
    logger.info("Loaded %d OSS code seeds from %s", len(seeds), p)
    return seeds


def _make_oss_seeded_prompt(
    *,
    rng: random.Random,
    task_type: str,
    seed_entry: dict[str, Any],
    enable_cot: bool,
) -> PromptExample:
    """
    Build a prompt that includes a real code snippet from the OSS seed pool.
    Used for bugfix, refactor, and code_review tasks so the model operates
    on actual code rather than inventing both the problem and the solution.
    """
    code = seed_entry["code"]
    language = seed_entry["language"]
    norm_lang = language.lower()
    display_lang = {"python": "Python", "typescript": "TypeScript",
                    "javascript": "JavaScript"}.get(norm_lang, language)

    constraint = rng.choice(CONSTRAINTS)
    style_hint = rng.choice(STYLE_HINTS)

    if task_type == "bugfix":
        user = (
            f"The following {display_lang} code may contain subtle bugs or edge-case failures.\n\n"
            f"```{norm_lang}\n{code}\n```\n\n"
            f"Identify any bugs or edge-case failures, explain why they happen, "
            f"and provide a corrected version.\n\n"
            f"Requirements:\n- {constraint}\n\nStyle hint: {style_hint}\n"
        )
    elif task_type == "refactor":
        user = (
            f"Refactor the following {display_lang} code for production readiness. "
            f"Apply clean code principles, improve naming, structure, and error handling.\n\n"
            f"```{norm_lang}\n{code}\n```\n\n"
            f"Requirements:\n- {constraint}\n\nStyle hint: {style_hint}\n"
        )
    elif task_type == "code_review":
        user = (
            f"Review the following {display_lang} code. Point out any performance issues, "
            f"security concerns, or style problems. Provide improved versions where appropriate.\n\n"
            f"```{norm_lang}\n{code}\n```\n\n"
            f"Requirements:\n- {constraint}\n\nStyle hint: {style_hint}\n"
        )
    else:
        user = (
            f"Extend and improve the following {display_lang} code. "
            f"Add missing functionality, error handling, and documentation.\n\n"
            f"```{norm_lang}\n{code}\n```\n\n"
            f"Requirements:\n- {constraint}\n\nStyle hint: {style_hint}\n"
        )

    system_pool = SYSTEM_VARIANTS_COT if enable_cot else SYSTEM_VARIANTS
    provenance = {
        "source": "oss_seeded_v1",
        "oss_source": seed_entry.get("source", "unknown"),
        "oss_license": seed_entry.get("license", "unknown"),
    }

    return PromptExample(
        id=_new_id(),
        task_type=task_type,
        language=norm_lang,
        messages=[
            {"role": "system", "content": rng.choice(system_pool)},
            {"role": "user", "content": user},
        ],
        meta=provenance,
    )


# ---------------------------------------------------------------------------
# Templated prompt builder (original, now with CoT option)
# ---------------------------------------------------------------------------

def _make_templated_prompt(
    *,
    rng: random.Random,
    task_type: str,
    language: str,
    enable_cot: bool,
) -> PromptExample:
    norm_lang = language.lower()
    display_lang = {"python": "Python", "py": "Python",
                    "typescript": "TypeScript", "ts": "TypeScript",
                    "javascript": "JavaScript", "js": "JavaScript"}.get(norm_lang, language)

    topic = rng.choice(TOPICS)
    constraint = rng.choice(CONSTRAINTS)
    fmt = rng.choice(FORMATS)
    style_hint = rng.choice(STYLE_HINTS)

    if task_type == "completion":
        user = (
            f"Write a {display_lang} implementation for {topic}. "
            f"Please provide {fmt}.\n\n"
            f"Requirements:\n- {constraint}"
        )
    elif task_type == "tests":
        user = (
            f"Write comprehensive unit tests in {display_lang} for a hypothetical "
            f"{fmt} that handles {topic}. Use standard testing libraries.\n\n"
            f"Requirements:\n- {constraint}\n"
            f"- Include both positive and negative (error) test cases."
        )
    elif task_type == "explain":
        user = (
            f"Explain the best practices for implementing {topic} in {display_lang}.\n\n"
            f"Requirements:\n- Provide clear code examples.\n"
            f"- Discuss performance and security implications.\n- {constraint}"
        )
    elif task_type == "code_review":
        user = (
            f"What are common performance or security bugs developers make when writing "
            f"{display_lang} code for {topic}? Provide bad code examples and then the "
            f"corrected versions.\n\nRequirements:\n- {constraint}"
        )
    elif task_type == "bugfix":
        user = (
            f"Create a tricky debugging scenario relating to {topic} in {display_lang}. "
            f"Show the buggy code, explain why it fails under certain edge cases, "
            f"and then provide the fix.\n\nRequirements:\n- {constraint}"
        )
    elif task_type == "refactor":
        user = (
            f"Show an example of poorly structured, messy {display_lang} code that handles "
            f"{topic}. Then, demonstrate how you would refactor it for production use, "
            f"applying clean code principles.\n\nRequirements:\n- {constraint}"
        )
    else:
        user = (
            f"Write a complete, production-ready {display_lang} script for {topic}.\n\n"
            f"Requirements:\n- {constraint}"
        )

    user += f"\n\nStyle hint: {style_hint}\n"
    system_pool = SYSTEM_VARIANTS_COT if enable_cot else SYSTEM_VARIANTS

    return PromptExample(
        id=_new_id(),
        task_type=task_type,
        language=norm_lang,
        messages=[
            {"role": "system", "content": rng.choice(system_pool)},
            {"role": "user", "content": user},
        ],
        meta={"source": "templated_v1", "topic": topic, "constraint": constraint},
    )


# ---------------------------------------------------------------------------
# Main pool builder
# ---------------------------------------------------------------------------

_OSS_ELIGIBLE_TASKS = {"bugfix", "refactor", "code_review"}


def build_prompt_pool(
    *,
    total: int,
    task_mix: dict[str, float],
    languages: list[str],
    seed: int | None = 42,
    cot_ratio: float = 0.5,
    oss_seed_path: str | Path | None = None,
    oss_ratio: float = 0.5,
) -> list[PromptExample]:
    """
    Build the prompt pool.

    Args:
        total: Total number of prompts to generate.
        task_mix: Task type -> weight mapping.
        languages: Languages to sample from.
        seed: Random seed.
        cot_ratio: Fraction of prompts that request chain-of-thought reasoning
                   (0.0 = never, 1.0 = always).
        oss_seed_path: Path to a JSONL file with OSS code seeds. If provided,
                       eligible tasks (bugfix, refactor, code_review) will use
                       real code snippets at the oss_ratio rate.
        oss_ratio: Fraction of eligible tasks that use OSS seeds vs templates
                   (0.0 = all templated, 1.0 = all seeded when seeds available).
    """
    rng = random.Random(seed)
    languages = languages or ["python"]

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

    # Load OSS seeds if available
    oss_seeds = load_oss_seeds(oss_seed_path)
    oss_by_lang: dict[str, list[dict[str, Any]]] = {}
    for s in oss_seeds:
        lang_key = s["language"].lower()
        oss_by_lang.setdefault(lang_key, []).append(s)

    out: list[PromptExample] = []
    for _ in range(total):
        task = sample_task()
        lang = rng.choice(languages)
        enable_cot = rng.random() < cot_ratio

        use_oss = (
            task in _OSS_ELIGIBLE_TASKS
            and oss_seeds
            and rng.random() < oss_ratio
            and lang.lower() in oss_by_lang
        )

        if use_oss:
            seed_entry = rng.choice(oss_by_lang[lang.lower()])
            ex = _make_oss_seeded_prompt(
                rng=rng, task_type=task, seed_entry=seed_entry, enable_cot=enable_cot,
            )
        else:
            ex = _make_templated_prompt(
                rng=rng, task_type=task, language=lang, enable_cot=enable_cot,
            )
        out.append(ex)

    cot_count = sum(
        1 for e in out
        if any("<think>" in m.get("content", "") for m in e.messages)
    )
    oss_count = sum(1 for e in out if e.meta.get("source", "").startswith("oss_seeded"))
    logger.info(
        "Built prompt pool: total=%d, cot=%d (%.0f%%), oss_seeded=%d (%.0f%%)",
        len(out), cot_count, 100 * cot_count / max(len(out), 1),
        oss_count, 100 * oss_count / max(len(out), 1),
    )
    return out