# Code Model Fine-Tuning: DeepSeek-Coder for Web Backend

2-Stage Training Pipeline: SFT → GRPO

## Project Overview

Fine-tune DeepSeek-Coder-6.7B for React, Django, and FastAPI development with capabilities for code completion, explanation, and bug fixing.

## Training Pipeline

```
Data Collection → Instruction Dataset (10K) → SFT → GRPO → Production Model → vLLM Deployment
```

## Project Structure

```
code-model-finetuning/
├── notebooks/          # 6 training notebooks (data collection → deployment)
├── data/              # Raw, processed, and validation datasets
├── configs/           # SFT and GRPO hyperparameters
├── models/            # Training checkpoints
├── src/               # Utilities and helpers
└── requirements.txt
```

## Tech Stack

- Model: DeepSeek-Coder-6.7B with 4-bit QLoRA
- Training: TRL (SFTTrainer, GRPO), Transformers, PEFT
- Deployment: vLLM
- Monitoring: Weights & Biases

## Resource Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 4090 24GB | A100 80GB |
| RAM | 64GB | 128GB |
| Storage | 500GB SSD | 1TB NVMe |
| Time | 3-4 days | 1-2 weeks |

## Learning Approach

Hands-on implementation with exercises covering QLoRA setup, GRPO reward modeling, metrics calculation, and inference optimization.
