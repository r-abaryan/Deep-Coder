"""Tests for distil_glm5.config — config loading and type validation."""

from __future__ import annotations

from pathlib import Path

from src.distil_glm5.config import (
    AppConfig,
    DedupConfig,
    FiltersConfig,
    GenerationConfig,
    JudgeConfig,
    PathsConfig,
    PromptPoolConfig,
    TeacherConfig,
    load_config,
)

_BASE_YAML = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


class TestLoadConfig:
    def test_loads_without_error(self) -> None:
        cfg = load_config(_BASE_YAML)
        assert isinstance(cfg, AppConfig)

    def test_section_types(self) -> None:
        cfg = load_config(_BASE_YAML)
        assert isinstance(cfg.paths, PathsConfig)
        assert isinstance(cfg.teacher, TeacherConfig)
        assert isinstance(cfg.generation, GenerationConfig)
        assert isinstance(cfg.prompt_pool, PromptPoolConfig)
        assert isinstance(cfg.filters, FiltersConfig)
        assert isinstance(cfg.judge, JudgeConfig)
        assert isinstance(cfg.dedup, DedupConfig)

    def test_teacher_url_trailing_slash_stripped(self) -> None:
        cfg = load_config(_BASE_YAML)
        assert not cfg.teacher.base_url.endswith("/")

    def test_paths_are_path_objects(self) -> None:
        cfg = load_config(_BASE_YAML)
        assert isinstance(cfg.paths.out_dir, Path)
        assert isinstance(cfg.paths.prompts_jsonl, Path)
        assert isinstance(cfg.paths.raw_jsonl, Path)
        assert isinstance(cfg.paths.curated_jsonl, Path)
        assert isinstance(cfg.paths.export_dir, Path)

    def test_generation_defaults(self) -> None:
        cfg = load_config(_BASE_YAML)
        assert cfg.generation.temperature >= 0
        assert cfg.generation.max_tokens > 0

    def test_task_mix_sums_to_one(self) -> None:
        cfg = load_config(_BASE_YAML)
        total = sum(cfg.prompt_pool.task_mix.values())
        assert abs(total - 1.0) < 1e-6
