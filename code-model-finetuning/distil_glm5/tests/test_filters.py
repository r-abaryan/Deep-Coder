"""Tests for distil_glm5.filters — refusal detection, syntax checks, redaction, etc."""

from __future__ import annotations

import pytest

from src.distil_glm5.filters import (
    FilterResult,
    build_curated_row,
    extract_python_from_markdown,
    filter_example,
    looks_like_refusal,
    normalize_for_hash,
    python_syntax_valid,
    redact_obvious_secrets,
)


# ---------------------------------------------------------------------------
# looks_like_refusal
# ---------------------------------------------------------------------------


class TestLooksLikeRefusal:
    def test_empty_string(self) -> None:
        assert looks_like_refusal("") is True

    def test_whitespace_only(self) -> None:
        assert looks_like_refusal("   \n\t  ") is True

    @pytest.mark.parametrize(
        "text",
        [
            "I can't help with that",
            "I cannot help with that request.",
            "I can't comply with this.",
            "I am unable to assist you.",
            "I can't assist with that.",
            "I won't be able to do this.",
        ],
    )
    def test_refusal_patterns(self, text: str) -> None:
        assert looks_like_refusal(text) is True

    def test_normal_code(self) -> None:
        assert looks_like_refusal("def hello(): return 'world'") is False

    def test_refusal_substring_in_longer_text(self) -> None:
        assert looks_like_refusal("Sure! I can't help with that part, but here is code...") is True


# ---------------------------------------------------------------------------
# python_syntax_valid
# ---------------------------------------------------------------------------


class TestPythonSyntaxValid:
    def test_valid_code(self) -> None:
        assert python_syntax_valid("x = 1 + 2") is True

    def test_invalid_code(self) -> None:
        assert python_syntax_valid("def foo(") is False

    def test_empty_string(self) -> None:
        assert python_syntax_valid("") is True

    def test_multiline(self) -> None:
        code = "def greet(name):\n    return f'Hello {name}'"
        assert python_syntax_valid(code) is True


# ---------------------------------------------------------------------------
# extract_python_from_markdown
# ---------------------------------------------------------------------------


class TestExtractPythonFromMarkdown:
    def test_fenced_block(self) -> None:
        md = "Here is code:\n```python\nx = 1\n```\nDone."
        assert extract_python_from_markdown(md) == "x = 1"

    def test_no_fence(self) -> None:
        raw = "x = 1"
        assert extract_python_from_markdown(raw) == "x = 1"

    def test_empty(self) -> None:
        assert extract_python_from_markdown("") == ""

    def test_py_alias(self) -> None:
        md = "```py\nprint('hi')\n```"
        assert extract_python_from_markdown(md) == "print('hi')"


# ---------------------------------------------------------------------------
# filter_example
# ---------------------------------------------------------------------------


class TestFilterExample:
    _DEFAULTS = dict(
        task_type="completion",
        language="python",
        min_chars=10,
        max_chars=5000,
        drop_refusals=True,
        require_python_syntax_valid=True,
        require_ts_js_syntax_valid=False,
    )

    def test_keeps_good_output(self) -> None:
        fr = filter_example(output_text="def add(a, b): return a + b", **self._DEFAULTS)
        assert fr.keep is True
        assert fr.reasons == []

    def test_rejects_too_short(self) -> None:
        fr = filter_example(output_text="hi", **self._DEFAULTS)
        assert fr.keep is False
        assert "too_short" in fr.reasons

    def test_rejects_too_long(self) -> None:
        fr = filter_example(output_text="x" * 6000, **{**self._DEFAULTS, "max_chars": 5000})
        assert fr.keep is False
        assert "too_long" in fr.reasons

    def test_rejects_refusal(self) -> None:
        fr = filter_example(output_text="I can't help with that", **self._DEFAULTS)
        assert fr.keep is False
        assert "refusal_or_empty" in fr.reasons

    def test_rejects_invalid_python(self) -> None:
        fr = filter_example(output_text="def foo(   # broken", **self._DEFAULTS)
        assert fr.keep is False
        assert "python_syntax_invalid" in fr.reasons

    def test_skips_syntax_for_explain_task(self) -> None:
        fr = filter_example(
            output_text="def foo(   # broken but it's an explanation",
            **{**self._DEFAULTS, "task_type": "explain"},
        )
        # explain is not in code_tasks, so syntax is not checked.
        assert "python_syntax_invalid" not in fr.reasons


# ---------------------------------------------------------------------------
# normalize_for_hash
# ---------------------------------------------------------------------------


class TestNormalizeForHash:
    def test_collapses_whitespace(self) -> None:
        assert normalize_for_hash("a   b\n\nc") == "a b c"

    def test_strips_python_fence(self) -> None:
        result = normalize_for_hash("```python\nx=1\n```")
        assert "python" not in result.lower()

    def test_none(self) -> None:
        assert normalize_for_hash("") == ""


# ---------------------------------------------------------------------------
# redact_obvious_secrets
# ---------------------------------------------------------------------------


class TestRedactObviousSecrets:
    def test_openai_key(self) -> None:
        text = "my key is sk-abc123def456ghi789jkl012"
        out, changed = redact_obvious_secrets(text)
        assert "[REDACTED_API_KEY]" in out
        assert changed is True

    def test_github_token(self) -> None:
        text = "token ghp_abcdefghijklmnopqrstuvwx"
        out, changed = redact_obvious_secrets(text)
        assert "[REDACTED_GITHUB_TOKEN]" in out
        assert changed is True

    def test_no_secrets(self) -> None:
        text = "just normal code here"
        out, changed = redact_obvious_secrets(text)
        assert out == text
        assert changed is False


# ---------------------------------------------------------------------------
# build_curated_row
# ---------------------------------------------------------------------------


class TestBuildCuratedRow:
    def test_schema(self) -> None:
        prompt = {
            "id": "abc123",
            "task_type": "completion",
            "language": "python",
            "messages": [
                {"role": "system", "content": "You are an expert."},
                {"role": "user", "content": "Write hello world."},
            ],
            "meta": {"source": "templated_v1"},
        }
        row = build_curated_row(
            prompt_row=prompt,
            teacher_model="test-model",
            gen_params={"temperature": 0.6},
            output_text="print('hello')",
            raw_response={},
            filter_reasons=[],
            redacted=False,
        )
        assert row["schema_version"] == 1
        assert row["id"] == "abc123"
        assert row["teacher"] == "test-model"
        assert row["messages"][-1]["role"] == "assistant"
        assert row["messages"][-1]["content"] == "print('hello')"
        assert "quality" in row

    def test_judge_fields(self) -> None:
        prompt = {
            "id": "j1",
            "task_type": "bugfix",
            "language": "python",
            "messages": [{"role": "user", "content": "fix this"}],
            "meta": {},
        }
        row = build_curated_row(
            prompt_row=prompt,
            teacher_model="m",
            gen_params={},
            output_text="fixed",
            raw_response={},
            filter_reasons=[],
            redacted=False,
            judge_passed=True,
            judge_reasoning="looks good",
        )
        assert row["quality"]["judge_passed"] is True
        assert row["quality"]["judge_reasoning"] == "looks good"
