from __future__ import annotations

import ast
import re
from dataclasses import dataclass
import shutil
import subprocess
import tempfile
from typing import Any


_REFUSAL_PATTERNS = [
    r"\bI can[’']?t help with that\b",
    r"\bI cannot help with that\b",
    r"\bI can[’']?t comply\b",
    r"\bI cannot comply\b",
    r"\bI am unable to\b",
    r"\bI can't assist\b",
    r"\bI cannot assist\b",
    r"\bI won[’']?t be able to\b",
]


@dataclass(frozen=True)
class FilterResult:
    keep: bool
    reasons: list[str]


def looks_like_refusal(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    for pat in _REFUSAL_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    return False


def python_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _extract_code_block(text: str, langs: tuple[str, ...]) -> str:
    """
    If the model returns fenced code blocks, prefer the first matching block.
    Otherwise return original.
    """
    if not text:
        return ""
    lang_pattern = "|".join(re.escape(l) for l in langs)
    pattern = rf"```(?:{lang_pattern})?\s*\n([\s\S]*?)\n```"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def extract_python_from_markdown(text: str) -> str:
    return _extract_code_block(text, ("python", "py"))


_HAS_TSC = shutil.which("tsc") is not None
_HAS_NODE = shutil.which("node") is not None


def typescript_syntax_valid(code: str) -> bool:
    """
    Best-effort TS syntax check using `tsc --noEmit` if available.
    If tsc is not installed, returns True (no-op).
    """
    if not _HAS_TSC:
        return True
    with tempfile.NamedTemporaryFile("w", suffix=".ts", delete=False, encoding="utf-8") as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            ["tsc", "--noEmit", "--pretty", "false", tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.returncode == 0
    finally:
        try:
            tempfile.os.remove(tmp_path)
        except OSError:
            pass


def javascript_syntax_valid(code: str) -> bool:
    """
    Best-effort JS syntax check using Node's Function constructor if available.
    If node is not installed, returns True (no-op).
    """
    if not _HAS_NODE:
        return True
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [
                "node",
                "-e",
                (
                    "const fs=require('fs');"
                    "const p=process.argv[2];"
                    "const src=fs.readFileSync(p,'utf8');"
                    "new Function(src);"
                ),
                tmp_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.returncode == 0
    finally:
        try:
            tempfile.os.remove(tmp_path)
        except OSError:
            pass


def filter_example(
    *,
    task_type: str,
    language: str,
    output_text: str,
    min_chars: int,
    max_chars: int,
    drop_refusals: bool,
    require_python_syntax_valid: bool,
    require_ts_js_syntax_valid: bool,
) -> FilterResult:
    reasons: list[str] = []
    t = (output_text or "").strip()
    lang = (language or "").lower()

    if len(t) < min_chars:
        reasons.append("too_short")
    if len(t) > max_chars:
        reasons.append("too_long")
    if drop_refusals and looks_like_refusal(t):
        reasons.append("refusal_or_empty")

    # Only run ast.parse on clearly Python outputs and code-centric tasks.
    if lang == "python" and require_python_syntax_valid and task_type in {
        "completion",
        "bugfix",
        "refactor",
    }:
        code = extract_python_from_markdown(t)
        if not python_syntax_valid(code):
            reasons.append("python_syntax_invalid")

    # Optional TS/JS syntax checks, if configured and toolchains are present.
    if lang in {"typescript", "ts"} and require_ts_js_syntax_valid and task_type in {
        "completion",
        "bugfix",
        "refactor",
    }:
        ts_code = _extract_code_block(t, ("ts", "typescript"))
        if not typescript_syntax_valid(ts_code):
            reasons.append("ts_syntax_invalid")

    if lang in {"javascript", "js"} and require_ts_js_syntax_valid and task_type in {
        "completion",
        "bugfix",
        "refactor",
    }:
        js_code = _extract_code_block(t, ("js", "javascript"))
        if not javascript_syntax_valid(js_code):
            reasons.append("js_syntax_invalid")

    return FilterResult(keep=len(reasons) == 0, reasons=reasons)


def normalize_for_hash(text: str) -> str:
    """
    Normalization used for exact dedup hashes.
    Keeps it intentionally simple (whitespace + common markdown fences).
    """
    t = (text or "").strip()
    t = re.sub(r"```(?:python)?", "```", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def redact_obvious_secrets(text: str) -> tuple[str, bool]:
    """
    Minimal redaction pass (heuristic). For release-grade redaction, add a dedicated PII model.
    Returns (redacted_text, changed).
    """
    patterns: list[tuple[str, str]] = [
        (r"(?i)\bsk-[a-z0-9]{20,}\b", "[REDACTED_API_KEY]"),
        (r"(?i)\bghp_[a-z0-9]{20,}\b", "[REDACTED_GITHUB_TOKEN]"),
        (r"(?i)\bAIza[0-9A-Za-z\-_]{30,}\b", "[REDACTED_API_KEY]"),
    ]
    out = text
    changed = False
    for pat, repl in patterns:
        new = re.sub(pat, repl, out)
        if new != out:
            changed = True
            out = new
    return out, changed


def build_curated_row(
    *,
    prompt_row: dict[str, Any],
    teacher_model: str,
    gen_params: dict[str, Any],
    output_text: str,
    raw_response: dict[str, Any],
    filter_reasons: list[str],
    redacted: bool,
    judge_passed: bool | None = None,
) -> dict[str, Any]:
    quality: dict[str, Any] = {"filter_reasons": filter_reasons, "redacted": redacted}
    if judge_passed is not None:
        quality["judge_passed"] = judge_passed
    return {
        "schema_version": 1,
        "id": prompt_row["id"],
        "task_type": prompt_row["task_type"],
        "language": prompt_row["language"],
        "messages": prompt_row["messages"] + [{"role": "assistant", "content": output_text}],
        "teacher": teacher_model,
        "gen": gen_params,
        "quality": quality,
        "provenance": prompt_row.get("meta", {}),
        "raw": {"teacher_response": raw_response},
    }

