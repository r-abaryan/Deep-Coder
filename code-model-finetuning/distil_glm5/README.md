## DISTIL_GLM5

Generate a code-focused distillation dataset by running a teacher model (`zai-org/GLM-5-FP8`) on a prompt pool, then filtering, deduplicating, and optionally judge-reviewing the results.

### Features

- **Chain-of-thought (CoT) prompting** — configurable fraction of prompts request `<think>` reasoning traces from the teacher, so the student can learn both reasoning and direct code generation.
- **OSS code seeding** — optionally seed bugfix/refactor/code_review tasks with real code snippets from permissively licensed repositories, so the teacher operates on actual code rather than inventing both the problem and the solution.
- **Task-aware judge/reviewer** — optional second model (e.g. Qwen-3.5 70B) evaluates teacher outputs with task-specific rubrics (correctness, edge cases, style) rather than a blanket YES/NO.
- **Incremental checkpointing** — generation and filtering scripts write partial progress to disk, so crashes don't lose hours of work. Generation supports `--resume`.
- **Per-row error handling** — individual generation failures are logged and skipped rather than crashing the entire run.

### Quickstart

From `code-model-finetuning/distil_glm5`:

```bash
python -m pip install -e ".[dev]"
```

Build prompts:

```bash
python scripts/build_prompt_pool.py --config configs/base.yaml
```

Generate teacher answers (against your vLLM/SGLang endpoint):

```bash
python scripts/generate_teacher_answers.py --config configs/base.yaml

# If interrupted, resume without re-generating completed prompts:
python scripts/generate_teacher_answers.py --config configs/base.yaml --resume
```

Filter + deduplicate (+ optional judge):

```bash
python scripts/filter_and_dedup.py --config configs/base.yaml
```

Export + report:

```bash
python scripts/export_dataset.py --config configs/base.yaml
python scripts/dataset_report.py --config configs/base.yaml
```

### OSS Code Seeds

To use real code snippets for bugfix/refactor/code_review tasks, prepare a JSONL file:

```jsonl
{"language": "python", "code": "def foo(x):\n    return x + 1\n", "source": "the-stack-v2", "license": "MIT"}
{"language": "typescript", "code": "function bar(s: string) { return s.trim(); }", "source": "my-repo", "license": "Apache-2.0"}
```

Then set in your config:

```yaml
prompt_pool:
  oss_seed_path: "data/oss_seeds.jsonl"
  oss_ratio: 0.5   # 50% of eligible tasks use real code
```

### Teacher serving (cloud)

Notes:
- `https://huggingface.co/zai-org/GLM-5-FP8`
- `https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html`

### Outputs

Default paths (configurable) are under:
- `out/prompts/` — prompt pool JSONL
- `out/raw/` — teacher raw generations JSONL
- `out/curated/` — filtered + deduped + judge-reviewed JSONL
- `out/export/` — final export artifacts

### Release checklist

- Drop secrets/keys/tokens (automated redaction in filter pipeline).
- Redact obvious PII where possible.
- Deduplicate (exact + near-duplicate).
- Track provenance and licenses for any OSS-grounded prompts.