## Getting started

This document explains how to:

1. Install dependencies
2. Serve GLM-5-FP8 with vLLM
3. Run the distillation scripts
4. Prepare data for GLM-4-9B SFT

---

### 1. Install dependencies

From the project root:

```bash
cd code-model-finetuning/distil_glm5
python -m pip install -e ".[hf,dev]"
```

If you want TypeScript/JavaScript syntax checks during filtering, install Node and TypeScript on the machine where you run the scripts:

```bash
node -v   # should print a version
tsc -v    # should print a version
```

These tools are optional. If they are missing, TS/JS syntax checks are skipped; other filters still run.

---

### 2. Serve GLM-5-FP8 with vLLM (cloud GPU)

Requires 8x H200 (or equivalent ~860 GB+ VRAM). On a Linux GPU machine:

Check your CUDA version first — the wheel URL must match:

```bash
nvcc --version   # e.g. 12.4 → cu124, 12.6 → cu126
```

```bash
uv pip install --upgrade --force-reinstall vllm --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly/cu124
uv pip install --upgrade --force-reinstall git+https://github.com/huggingface/transformers.git
uv pip install --force-reinstall numba
```

> GLM-5 uses `glm_moe_dsa` architecture which requires a bleeding-edge transformers build.
> The git install above is mandatory — released PyPI versions will fail with
> `"Transformers does not recognize this architecture"`.

Start the server:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

vllm serve zai-org/GLM-5-FP8 \
  --served-model-name "zai-org/GLM-5-FP8" \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.93 \
  --max_num_batched_tokens 4096 \
  --max-model-len 16384 \
  --enforce-eager \
  --seed 3407
```

Flag notes:
- **No `--kv-cache-dtype fp8`** — GLM-5's MLA attention (`head_size=576`, `qk_nope_head_dim=192`) is incompatible with FP8 KV cache in vLLM. All MLA attention backends reject `kv_cache_dtype=fp8` for this architecture. With 8x H200 (~1.4TB VRAM) the default FP16 KV cache is fine.
- **`--enforce-eager`** — disables CUDA graph capture. Required because GLM-5's `FLASHMLA_SPARSE` attention backend uses a `sparse_attn_indexer` op that calls `.item()` during forward, which is illegal during CUDA graph capture. Throughput is slightly lower without CUDA graphs but the model runs correctly.
- **No `--speculative-config.method mtp`** — MTP speculation combined with sparse MLA attention triggers the same CUDA graph capture failure. Omitted until vLLM fixes the incompatibility.
- `--max-model-len 16384` is enough for our prompts (max_tokens is 2048); no need for the full 200K context.

vLLM exposes an OpenAI-compatible API at `http://<host>:8000/v1`. Leave this process running while you generate data.

---

### 3. Configure `base.yaml`

Edit `configs/base.yaml` with the correct endpoint:

```yaml
teacher:
  model_id: "zai-org/GLM-5-FP8"
  base_url: "http://<cloud_host_or_ip>:8000/v1"
  api_key: "EMPTY"
  timeout_s: 120
  max_retries: 4
  concurrency: 16

prompt_pool:
  languages: ["python", "typescript", "javascript"]
  total_prompts: 2000

filters:
  min_output_chars: 64
  max_output_chars: 20000
  drop_refusals: true
  require_python_syntax_valid: true
  require_ts_js_syntax_valid: true
```

Adjust `total_prompts` and `concurrency` for your hardware.

Optional: a second model (judge) can accept or reject teacher outputs (rejection sampling). Off by default. In `configs/base.yaml`:

```yaml
judge:
  enabled: false
  model_id: "Qwen/Qwen3.5-70B-Instruct"
  base_url: "http://localhost:8001/v1"
  api_key: "EMPTY"
  timeout_s: 90
  max_retries: 2
  max_tokens: 256
  temperature: 0.0
  concurrency: 8
```

Set `enabled: true` and point `base_url` at a server that serves the judge model. The judge needs 256 tokens to write a brief task-specific evaluation + YES/NO verdict. It runs during `filter_and_dedup.py`; only rows it accepts are kept.

> The judge requires a **separate GPU endpoint** — GLM-5-FP8 at TP=8 uses all GPUs. Either use a second node, or run sequentially: generate with GLM-5, kill the server, then serve the judge model on the same GPUs before running filter_and_dedup.

---

### 4. Run the distillation pipeline

All commands below are from:

```bash
cd code-model-finetuning/distil_glm5
```

#### 4.1 Build prompt pool

```bash
python scripts/build_prompt_pool.py --config configs/base.yaml
```

Output:

- `out/prompts/prompt_pool.jsonl`

#### 4.2 Generate teacher answers

Make sure the vLLM server is running and reachable, then run:

```bash
python scripts/generate_teacher_answers.py --config configs/base.yaml
```

Output:

- `out/raw/teacher_generations.jsonl`

#### 4.3 Filter and deduplicate

```bash
python scripts/filter_and_dedup.py --config configs/base.yaml
```

This step:

- drops very short / very long answers
- drops refusals / empty answers
- validates Python, TS, and JS syntax (if the relevant tools are available)
- removes exact duplicates

Output:

- `out/curated/curated.jsonl`

#### 4.4 Export and inspect

```bash
python scripts/export_dataset.py --config configs/base.yaml --format jsonl
python scripts/dataset_report.py --config configs/base.yaml
```

The JSONL export writes `out/export/train.jsonl` (curated rows with the bulky `raw` field stripped) and `out/export/meta.json`.

The report shows counts by task type, language, CoT ratio, and difficulty.

If you want a Hugging Face dataset on disk:

```bash
python scripts/export_dataset.py --config configs/base.yaml --format hf
```

---

### 4.5 Run tests (optional)

```bash
python -m pytest tests/ -v
```

Tests cover filters, judge verdict parsing, prompt generation, and config loading. No GPU or API access needed.

---

### 5. Prepare data for GLM-4-9B SFT (outline)

This section sketches how to consume `curated.jsonl` for GLM-4-9B SFT. It is not wired into this folder yet; you can turn it into a separate script or notebook.

1. Install training stack:

```bash
python -m pip install "transformers>=4.36" "datasets>=2.16" "trl>=0.7.0" "accelerate>=0.25.0" "peft>=0.7.0"
```

2. Load tokenizer and model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "zai-org/GLM-4-9B-0414"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
```

3. Convert `messages` to text with GLM-4-9B chat template:

```python
from datasets import load_dataset

ds = load_dataset(
    "json",
    data_files="code-model-finetuning/distil_glm5/out/curated/curated.jsonl",
    split="train",
)

def format_example(ex):
    text = tokenizer.apply_chat_template(
        ex["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

ds_formatted = ds.map(format_example, remove_columns=ds.column_names)
```

4. Run SFT with TRL:

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="glm4_distilled_sft",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,
    logging_steps=50,
    save_steps=1000,
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_formatted,
)

trainer.train()
trainer.save_model("glm4_distilled_sft")
```
