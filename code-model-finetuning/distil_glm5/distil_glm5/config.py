from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _p(v: str | Path) -> Path:
    return v if isinstance(v, Path) else Path(v)


@dataclass(frozen=True)
class TeacherConfig:
    model_id: str
    base_url: str
    api_key: str
    timeout_s: float
    max_retries: int
    concurrency: int


@dataclass(frozen=True)
class GenerationConfig:
    seed: int | None
    temperature: float
    top_p: float
    max_tokens: int
    n: int
    stop: list[str]


@dataclass(frozen=True)
class PathsConfig:
    out_dir: Path
    prompts_jsonl: Path
    raw_jsonl: Path
    curated_jsonl: Path
    export_dir: Path


@dataclass(frozen=True)
class PromptPoolConfig:
    languages: list[str]
    total_prompts: int
    task_mix: dict[str, float]
    cot_ratio: float
    oss_seed_path: str | None
    oss_ratio: float


@dataclass(frozen=True)
class FiltersConfig:
    min_output_chars: int
    max_output_chars: int
    drop_refusals: bool
    require_python_syntax_valid: bool
    require_ts_js_syntax_valid: bool


@dataclass(frozen=True)
class JudgeConfig:
    enabled: bool
    model_id: str
    base_url: str
    api_key: str
    timeout_s: float
    max_retries: int
    max_tokens: int
    temperature: float
    concurrency: int


@dataclass(frozen=True)
class DedupConfig:
    enable_exact: bool
    enable_near: bool
    near_jaccard_threshold: float


@dataclass(frozen=True)
class AppConfig:
    run_name: str
    paths: PathsConfig
    teacher: TeacherConfig
    generation: GenerationConfig
    prompt_pool: PromptPoolConfig
    filters: FiltersConfig
    judge: JudgeConfig
    dedup: DedupConfig


def load_config(config_path: str | Path) -> AppConfig:
    config_path = _p(config_path)
    raw: dict[str, Any]
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    paths_raw = raw["paths"]
    paths = PathsConfig(
        out_dir=_p(paths_raw["out_dir"]),
        prompts_jsonl=_p(paths_raw["prompts_jsonl"]),
        raw_jsonl=_p(paths_raw["raw_jsonl"]),
        curated_jsonl=_p(paths_raw["curated_jsonl"]),
        export_dir=_p(paths_raw["export_dir"]),
    )

    teacher_raw = raw["teacher"]
    teacher = TeacherConfig(
        model_id=str(teacher_raw["model_id"]),
        base_url=str(teacher_raw["base_url"]).rstrip("/"),
        api_key=str(teacher_raw["api_key"]),
        timeout_s=float(teacher_raw.get("timeout_s", 120)),
        max_retries=int(teacher_raw.get("max_retries", 4)),
        concurrency=int(teacher_raw.get("concurrency", 16)),
    )

    gen_raw = raw["generation"]
    generation = GenerationConfig(
        seed=None if gen_raw.get("seed", None) is None else int(gen_raw["seed"]),
        temperature=float(gen_raw.get("temperature", 0.6)),
        top_p=float(gen_raw.get("top_p", 0.95)),
        max_tokens=int(gen_raw.get("max_tokens", 2048)),
        n=int(gen_raw.get("n", 1)),
        stop=list(gen_raw.get("stop", [])),
    )

    pp_raw = raw["prompt_pool"]
    prompt_pool = PromptPoolConfig(
        languages=list(pp_raw.get("languages", ["python"])),
        total_prompts=int(pp_raw.get("total_prompts", 2000)),
        task_mix={k: float(v) for k, v in dict(pp_raw.get("task_mix", {})).items()},
        cot_ratio=float(pp_raw.get("cot_ratio", 0.5)),
        oss_seed_path=pp_raw.get("oss_seed_path", None),
        oss_ratio=float(pp_raw.get("oss_ratio", 0.5)),
    )

    filt_raw = raw["filters"]
    filters = FiltersConfig(
        min_output_chars=int(filt_raw.get("min_output_chars", 64)),
        max_output_chars=int(filt_raw.get("max_output_chars", 20000)),
        drop_refusals=bool(filt_raw.get("drop_refusals", True)),
        require_python_syntax_valid=bool(filt_raw.get("require_python_syntax_valid", True)),
        require_ts_js_syntax_valid=bool(filt_raw.get("require_ts_js_syntax_valid", True)),
    )

    judge_raw = raw.get("judge", {})
    judge = JudgeConfig(
        enabled=bool(judge_raw.get("enabled", False)),
        model_id=str(judge_raw.get("model_id", "Qwen/Qwen3.5-70B-Instruct")),
        base_url=str(judge_raw.get("base_url", "http://localhost:8001/v1")).rstrip("/"),
        api_key=str(judge_raw.get("api_key", "EMPTY")),
        timeout_s=float(judge_raw.get("timeout_s", 90)),
        max_retries=int(judge_raw.get("max_retries", 2)),
        max_tokens=int(judge_raw.get("max_tokens", 256)),
        temperature=float(judge_raw.get("temperature", 0.0)),
        concurrency=int(judge_raw.get("concurrency", 8)),
    )

    dedup_raw = raw.get("dedup", {})
    dedup = DedupConfig(
        enable_exact=bool(dedup_raw.get("enable_exact", True)),
        enable_near=bool(dedup_raw.get("enable_near", False)),
        near_jaccard_threshold=float(dedup_raw.get("near_jaccard_threshold", 0.92)),
    )

    return AppConfig(
        run_name=str(raw.get("run_name", config_path.stem)),
        paths=paths,
        teacher=teacher,
        generation=generation,
        prompt_pool=prompt_pool,
        filters=filters,
        judge=judge,
        dedup=dedup,
    )