"""Tests for distil_glm5.judge — verdict parsing and instruction extraction."""

from __future__ import annotations

import pytest

from src.distil_glm5.judge import _parse_verdict, get_instruction_from_row


# ---------------------------------------------------------------------------
# _parse_verdict
# ---------------------------------------------------------------------------


class TestParseVerdict:
    def test_yes_on_last_line(self) -> None:
        text = "The code is correct and handles edge cases well.\nYES"
        keep, reasoning = _parse_verdict(text)
        assert keep is True
        assert "correct" in reasoning

    def test_no_on_last_line(self) -> None:
        text = "Missing error handling for empty input.\nNO"
        keep, reasoning = _parse_verdict(text)
        assert keep is False
        assert "Missing" in reasoning

    def test_empty_response(self) -> None:
        keep, reasoning = _parse_verdict("")
        assert keep is False
        assert reasoning == ""

    def test_yes_fallback_scan(self) -> None:
        text = "Overall YES, this is good code.\nFinal thoughts above."
        keep, _ = _parse_verdict(text)
        assert keep is True

    def test_no_fallback_when_ambiguous(self) -> None:
        text = "Some generic feedback with no verdict."
        keep, _ = _parse_verdict(text)
        assert keep is False

    def test_case_insensitivity(self) -> None:
        text = "Good code.\nyes"
        keep, _ = _parse_verdict(text)
        assert keep is True


# ---------------------------------------------------------------------------
# get_instruction_from_row
# ---------------------------------------------------------------------------


class TestGetInstructionFromRow:
    def test_extracts_user_message(self) -> None:
        row = {
            "messages": [
                {"role": "system", "content": "You are an expert."},
                {"role": "user", "content": "Write a sort function."},
            ]
        }
        assert get_instruction_from_row(row) == "Write a sort function."

    def test_takes_last_user_message(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Answer"},
                {"role": "user", "content": "Follow-up question"},
            ]
        }
        assert get_instruction_from_row(row) == "Follow-up question"

    def test_no_messages(self) -> None:
        assert get_instruction_from_row({}) == ""

    def test_no_user_role(self) -> None:
        row = {"messages": [{"role": "system", "content": "sys"}]}
        assert get_instruction_from_row(row) == ""
