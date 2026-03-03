"""Tests for distil_glm5.prompts — prompt pool generation and OSS seed loading."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from distil_glm5.prompts import PromptExample, build_prompt_pool, load_oss_seeds


# ---------------------------------------------------------------------------
# build_prompt_pool
# ---------------------------------------------------------------------------


class TestBuildPromptPool:
    def test_correct_count(self) -> None:
        pool = build_prompt_pool(
            total=20,
            task_mix={"completion": 0.5, "bugfix": 0.5},
            languages=["python"],
            seed=42,
        )
        assert len(pool) == 20

    def test_deterministic_with_seed(self) -> None:
        kwargs = dict(total=10, task_mix={"completion": 1.0}, languages=["python"], seed=99)
        a = build_prompt_pool(**kwargs)
        b = build_prompt_pool(**kwargs)
        assert [e.id for e in a] == [e.id for e in b]

    def test_all_entries_are_prompt_examples(self) -> None:
        pool = build_prompt_pool(
            total=5, task_mix={"explain": 1.0}, languages=["typescript"], seed=1,
        )
        for ex in pool:
            assert isinstance(ex, PromptExample)
            assert ex.language == "typescript"
            assert ex.task_type == "explain"

    def test_cot_ratio_all(self) -> None:
        pool = build_prompt_pool(
            total=10, task_mix={"completion": 1.0}, languages=["python"],
            seed=7, cot_ratio=1.0,
        )
        for ex in pool:
            system_content = ex.messages[0]["content"]
            assert "<think>" in system_content

    def test_cot_ratio_none(self) -> None:
        pool = build_prompt_pool(
            total=10, task_mix={"completion": 1.0}, languages=["python"],
            seed=7, cot_ratio=0.0,
        )
        for ex in pool:
            system_content = ex.messages[0]["content"]
            assert "<think>" not in system_content

    def test_multilang(self) -> None:
        pool = build_prompt_pool(
            total=60, task_mix={"completion": 1.0},
            languages=["python", "typescript", "javascript"], seed=42,
        )
        langs = {ex.language for ex in pool}
        assert langs == {"python", "typescript", "javascript"}

    def test_to_json_schema_version(self) -> None:
        pool = build_prompt_pool(
            total=1, task_mix={"completion": 1.0}, languages=["python"], seed=0,
        )
        j = pool[0].to_json()
        assert j["schema_version"] == 1
        assert "messages" in j


# ---------------------------------------------------------------------------
# load_oss_seeds
# ---------------------------------------------------------------------------


class TestLoadOssSeeds:
    def test_none_path(self) -> None:
        assert load_oss_seeds(None) == []

    def test_missing_file(self, tmp_path: Path) -> None:
        assert load_oss_seeds(tmp_path / "nope.jsonl") == []

    def test_valid_file(self, tmp_path: Path) -> None:
        p = tmp_path / "seeds.jsonl"
        rows = [
            {"language": "python", "code": "x = 1", "source": "test", "license": "MIT"},
            {"language": "typescript", "code": "let x = 1;", "source": "test", "license": "MIT"},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        seeds = load_oss_seeds(p)
        assert len(seeds) == 2

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "seeds.jsonl"
        p.write_text(
            '{"language": "python", "code": "ok"}\n'
            "not json\n"
            '{"no_code_key": true}\n',
            encoding="utf-8",
        )
        seeds = load_oss_seeds(p)
        assert len(seeds) == 1
