## DISTIL_GLM5

Generate a code-focused distillation dataset by running a teacher model (planned: `zai-org/GLM-5-FP8`) on a prompt pool, (optionaly 2nd Model as reviewer can be enabled for Gold standard produced Dataset), then filtering/deduping and exporting the results.

### Quickstart

From `code-model-finetuning/distil_glm5`:

```bash
python -m pip install -e ".[dev]"
```

Build prompts:

```bash
python scripts/build_prompt_pool.py --config configs/base.yaml
```

Generate teacher answers (later, against your cloud endpoint):

```bash
python scripts/generate_teacher_answers.py --config configs/base.yaml
```

Filter + deduplicate:

```bash
python scripts/filter_and_dedup.py --config configs/base.yaml
```

Export + report:

```bash
python scripts/export_dataset.py --config configs/base.yaml
python scripts/dataset_report.py --config configs/base.yaml
```

### Teacher serving (cloud)

Notes:
- `https://huggingface.co/zai-org/GLM-5-FP8`
- `https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html`

### Outputs

Default paths (configurable) are under:
- `distil_glm5/out/prompts/` (prompt pool JSONL)
- `distil_glm5/out/raw/` (teacher raw generations JSONL)
- `distil_glm5/out/curated/` (filtered+deduped JSONL)
- `distil_glm5/out/export/` (final export artifacts)

### Release checklist

- Drop secrets/keys/tokens.
- Redact obvious PII where possible.
- Deduplicate (exact + near-duplicate).
- Track provenance and licenses for any OSS-grounded prompts.

