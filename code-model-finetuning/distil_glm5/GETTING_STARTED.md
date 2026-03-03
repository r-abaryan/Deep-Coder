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

On a Linux GPU machine (A100/H100 or similar):

```bash
python -m pip install -U "vllm[all]" --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly
python -m pip install "git+https://github.com/huggingface/transformers.git"
```

Start the server:

```bash
vllm serve zai-org/GLM-5-FP8 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --served-model-name "zai-org/GLM-5-FP8"
```

By default vLLM exposes an OpenAI-compatible API at:

- `http://<host>:8000/v1`

Leave this process running while you generate data.

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
  model_id: "Qwen/Qwen2.5-Coder-7B-Instruct"
  base_url: "http://localhost:8000/v1"
  api_key: "EMPTY"
  timeout_s: 60
  max_retries: 2
  max_tokens: 10
  temperature: 0.0
```

Set `enabled: true` and point `base_url` at a server that serves the judge model (e.g. same vLLM with a second model, or another endpoint). The judge is called during `filter_and_dedup.py`; only rows it accepts are kept.

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

You can adapt this outline into a dedicated training script or notebook under `code-model-finetuning`.

