from __future__ import annotations

import ast
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any


_REFUSAL_PATTERNS = [
    r"\bI can['']?t help with that\b",
    r"\bI cannot help with that\b",
    r"\bI can['']?t comply\b",
    r"\bI cannot comply\b",
    r"\bI am unable to\b",
    r"\bI can't assist\b",
    r"\bI cannot assist\b",
    r"\bI won['']?t be able to\b",
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
    lang_pattern = "|".join(re.escape(lang) for lang in langs)
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
    Ignores module resolution and type-checking errors (TS2xxx/TS7xxx),
    failing only if there are syntax errors (TS1xxx).
    """
    if not _HAS_TSC:
        return True
    with tempfile.NamedTemporaryFile("w", suffix=".ts", delete=False, encoding="utf-8") as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            ["tsc", "--noEmit", "--skipLibCheck", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        out = proc.stdout + "\n" + proc.stderr
        if re.search(r"TS1\d{3}:", out):
            return False
        return True
    finally:
        try:
            tempfile.os.remove(tmp_path)  # type: ignore[attr-defined]
        except OSError:
            pass


def javascript_syntax_valid(code: str) -> bool:
    """
    Best-effort JS syntax check using Node's syntax check if available.
    Supports ES modules.
    """
    if not _HAS_NODE:
        return True
    try:
        proc = subprocess.run(
            ["node", "--check", "--input-type=module"],
            input=code.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.returncode == 0
    except OSError:
        return True


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

    code_tasks = {"completion", "bugfix", "refactor"}

    if lang == "python" and require_python_syntax_valid and task_type in code_tasks:
        code = extract_python_from_markdown(t)
        if not python_syntax_valid(code):
            reasons.append("python_syntax_invalid")

    if lang in {"typescript", "ts"} and require_ts_js_syntax_valid and task_type in code_tasks:
        ts_code = _extract_code_block(t, ("ts", "typescript"))
        if not typescript_syntax_valid(ts_code):
            reasons.append("ts_syntax_invalid")

    if lang in {"javascript", "js"} and require_ts_js_syntax_valid and task_type in code_tasks:
        js_code = _extract_code_block(t, ("js", "javascript"))
        if not javascript_syntax_valid(js_code):
            reasons.append("js_syntax_invalid")

    return FilterResult(keep=len(reasons) == 0, reasons=reasons)


def normalize_for_hash(text: str) -> str:
    """Normalization for exact dedup hashes."""
    t = (text or "").strip()
    t = re.sub(r"```(?:python|py|typescript|ts|javascript|js)?", "```", t, flags=re.IGNORECASE)
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


def get_difficulty_score(prompt_row: dict[str, Any], output_text: str) -> dict[str, Any]:
    """Calculate basic difficulty metrics based on length and syntactical complexity."""
    t = output_text.strip()
    score: dict[str, Any] = {
        "length": len(t),
        "lines": len(t.splitlines()),
        "has_comments": bool(re.search(r'#|//|/\*|"""', t)),
        "has_types": bool(re.search(r':\s*[A-Z][a-zA-Z]+', t)),
        "import_diversity": len(set(re.findall(r'import (\w+)|from (\w+)', t))),
    }

    if score["lines"] < 15 and not score["has_types"]:
        score["level"] = "basic"
    elif score["lines"] > 50 and score["has_types"] and score["import_diversity"] > 2:
        score["level"] = "advanced"
    else:
        score["level"] = "intermediate"

    return score


def build_minhash(text: str, num_perm: int = 128) -> Any:
    """Build a datasketch MinHash object for near-dedup. Returns None if datasketch missing."""
    try:
        from datasketch import MinHash  # type: ignore
    except ImportError:
        return None

    m = MinHash(num_perm=num_perm)
    for token in text.split():
        m.update(token.encode("utf-8"))
    return m


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
    judge_reasoning: str | None = None,
) -> dict[str, Any]:
    difficulty = get_difficulty_score(prompt_row, output_text)
    quality: dict[str, Any] = {
        "filter_reasons": filter_reasons,
        "redacted": redacted,
        "difficulty": difficulty,
    }
    if judge_passed is not None:
        quality["judge_passed"] = judge_passed
    if judge_reasoning is not None:
        quality["judge_reasoning"] = judge_reasoning
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