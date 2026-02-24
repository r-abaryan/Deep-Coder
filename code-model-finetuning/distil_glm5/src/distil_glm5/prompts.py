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
    norm_lang = language.lower()
    if norm_lang in {"python", "py"}:
        language = "Python"
    elif norm_lang in {"typescript", "ts"}:
        language = "TypeScript"
    elif norm_lang in {"javascript", "js"}:
        language = "JavaScript"
        
    topics = [
        "string manipulation", "data validation", "API request handling", "database schema querying", 
        "data serialization", "matrix operations", "regex parsing", "file reading and writing",
        "graph traversal", "binary tree manipulation", "sorting arrays", "hashing data",
        "encryption/decryption wrappers", "user authentication logic", "logging middleware",
        "background job processing", "HTML parsing", "CLI argument parsing", "image processing",
        "rate limiting", "websocket communication", "caching mechanisms", "retry logic with exponential backoff",
        "date and timezone manipulation", "concurrent task execution", "configuration management",
        "state management", "custom array iteration", "mathematical computations", "memory caching"
    ]
    constraints = [
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
        "Use idiomatic patterns standard for this language."
    ]
    formats = [
        "a single function",
        "a class with accompanying methods",
        "a small utility module",
        "a set of purely functional helpers"
    ]
    
    topic = rng.choice(topics)
    constraint = rng.choice(constraints)
    fmt = rng.choice(formats)
    
    if task_type == "completion":
        user = f"Write a {language} implementation for {topic}. Please provide {fmt}.\n\nRequirements:\n- {constraint}"
    elif task_type == "tests":
        user = f"Write comprehensive unit tests in {language} for a hypothetical {fmt} that handles {topic}. Use standard testing libraries.\n\nRequirements:\n- {constraint}\n- Include both positive and negative (error) test cases."
    elif task_type == "explain":
        user = f"Explain the best practices for implementing {topic} in {language}.\n\nRequirements:\n- Provide clear code examples.\n- Discuss performance and security implications.\n- {constraint}"
    elif task_type == "code_review":
        user = f"What are common performance or security bugs developers make when writing {language} code for {topic}? Provide bad code examples and then the corrected versions.\n\nRequirements:\n- {constraint}"
    elif task_type == "bugfix":
        user = f"Create a tricky debugging scenario relating to {topic} in {language}. Show the buggy code, explain why it fails under certain edge cases, and then provide the fix.\n\nRequirements:\n- {constraint}"
    elif task_type == "refactor":
        user = f"Show an example of poorly structured, messy {language} code that handles {topic}. Then, demonstrate how you would refactor it for production use, applying clean code principles.\n\nRequirements:\n- {constraint}"
    else:
        user = f"Write a complete, production-ready {language} script for {topic}.\n\nRequirements:\n- {constraint}"

    style_hint = rng.choice(
        [
            "Prefer clear variable names.",
            "Write modular code.",
            "Use comprehensive type annotations where applicable.",
            "Make sure edge cases like null or empty inputs are handled.",
            "Minimize nesting by using early returns.",
            "Avoid magic numbers or hardcoded strings."
        ]
    )
    user = user + f"\n\nStyle hint: {style_hint}\n"

    system_variants = [
        SYSTEM_CODE,
        "You are a senior software developer with years of experience. Provide clear, correct, and robust code.",
        "You are an expert programmer. Output high-quality code and minimal fluff.",
        "You are an AI coding assistant. Ensure your solutions are optimal and thoroughly commented."
    ]

    return PromptExample(
        id=_new_id(),
        task_type=task_type,
        language=language.lower(),
        messages=[
            {"role": "system", "content": rng.choice(system_variants)},
            {"role": "user", "content": user},
        ],
        meta={"source": "templated_v1", "topic": topic, "constraint": constraint},
    )

