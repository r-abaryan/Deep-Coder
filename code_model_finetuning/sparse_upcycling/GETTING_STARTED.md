# Sparse Upcycling: Qwen3-8B → MoE (4 experts)

## What is Sparse Upcycling?

A technique to convert a pre-trained **dense** transformer into a **Mixture-of-Experts (MoE)** model without training from scratch.

```
DENSE MODEL (Qwen3-8B)              MoE MODEL (~28B params, ~13B active)
┌──────────────────────┐            ┌──────────────────────────────────┐
│  Layer 0             │            │  Layer 0 (MoE)                   │
│  ┌────────────────┐  │            │  ┌────────┐                      │
│  │  Attention     │  │  ──copy──► │  │ Attn   │  (unchanged)         │
│  ├────────────────┤  │            │  ├────────┤                      │
│  │  FFN           │  │            │  │ Router │→  Expert 0 (FFN copy)│
│  │  (gate,up,down)│  │  ──×4───►  │  │        │→  Expert 1 (FFN copy)│
│  └────────────────┘  │            │  │        │→  Expert 2 (FFN copy)│
│                      │            │  │        │→  Expert 3 (FFN copy)│
│  Layer 1             │            │  ├───────────────────────────────┤
│  ┌────────────────┐  │            │  Layer 1 (DENSE - unchanged)     │
│  │  Attention     │  │  ──copy──► │  ┌────────────────┐              │
│  ├────────────────┤  │            │  │  Attention     │              │
│  │  FFN           │  │  ──copy──► │  ├────────────────┤              │
│  └────────────────┘  │            │  │  FFN (original)│              │
│  ...                 │            │  └────────────────┘              │
└──────────────────────┘            │  ...                             │
                                    └──────────────────────────────────┘
                                    Even layers = MoE, Odd layers = Dense
```

**Why alternating?** If every layer gets a random router at once, the model collapses. Dense "anchor" layers keep representations stable while routers learn.

**Why not train from scratch?** A 28B MoE from scratch needs trillions of tokens. Upcycling reuses the 8B model's knowledge and needs only ~2-5% of the original training compute.

## What are "Layer Indices"?

Qwen3-8B is a stack of 36 transformer layers, numbered 0 through 35. Each layer has:
- **Attention** (Q, K, V, O projections) - learns relationships between tokens
- **FFN / MLP** (gate_proj, up_proj, down_proj) - transforms each token's representation

"Even indices" means layers 0, 2, 4, 6, 8, ..., 34. "Odd indices" means 1, 3, 5, 7, ..., 35.

```
Layer 0  → MoE (even)     ← FFN replaced with 4 experts + router
Layer 1  → Dense (odd)    ← FFN stays original, untouched
Layer 2  → MoE (even)     ← FFN replaced with 4 experts + router
Layer 3  → Dense (odd)    ← FFN stays original, untouched
...
Layer 34 → MoE (even)     ← FFN replaced with 4 experts + router
Layer 35 → Dense (odd)    ← FFN stays original, untouched
```

18 even layers become MoE, 18 odd layers stay dense. The dense layers act as **anchors** - stable reference points that keep the model's representations coherent while the 18 random routers are still learning.

## What We Freeze and Why

"Freezing" a parameter means `requires_grad=False` - it doesn't receive gradients and doesn't change during training. This saves memory (no optimizer states needed) and prevents destroying already-learned knowledge.

```
┌─────────────────┬──────────┬──────────────────────────────────────┐
│ Component       │ Trainable│ Reason                               │
├─────────────────┼──────────┼──────────────────────────────────────┤
│ Embeddings      │ FROZEN   │ Token representations are already    │
│ (token + pos)   │          │ well-learned from pretraining.       │
│                 │          │ Changing them destabilizes everything│
│                 │          │ downstream.                          │
├─────────────────┼──────────┼──────────────────────────────────────┤
│ Attention       │ FROZEN   │ Q/K/V/O matrices learn relationships │
│ (Q, K, V, O)    │          │ between tokens - trained on trillions│
│                 │          │ of tokens already. MoE conversion    │
│                 │          │ only changes FFNs, not how tokens    │
│                 │          │ attend to each other.                │
├─────────────────┼──────────┼──────────────────────────────────────┤
│ Expert FFNs     │ TRAINED  │ The duplicated MLPs. They must       │
│ (gate,up,down)  │          │ DIFFERENTIATE - each expert learns   │
│                 │          │ to handle different token types.     │
│                 │          │ Full-param training gives them max   │
│                 │          │ freedom to diverge from each other.  │
├─────────────────┼──────────┼──────────────────────────────────────┤
│ Routers         │ TRAINED  │ Randomly initialized - must learn    │
│                 │          │ from scratch. Gets 5× higher LR      │
│                 │          │ because it starts from random while  │
│                 │          │ experts start from pretrained.       │
├─────────────────┼──────────┼──────────────────────────────────────┤
│ Dense FFNs      │ FROZEN   │ Odd-indexed layers keep original FFN │
│ (anchor layers) │          │ unchanged - stable anchors.          │
├─────────────────┼──────────┼──────────────────────────────────────┤
│ LayerNorm/RMS   │ TRAINED  │ Normalization must adapt to new MoE  │
│                 │          │ output distributions.                │
└─────────────────┴──────────┴──────────────────────────────────────┘
```

## Architecture Choices

| Parameter | Value | Why |
|---|---|---|
| Base model | Qwen3-8B | Strong dense model, SwiGLU FFN, GQA |
| Experts | 4 per MoE layer | Balance of capacity vs memory |
| Top-k | 2 | Each token uses 2 of 4 experts |
| MoE layers | 18 (even indices 0,2,...,34) | Alternating for stability |
| Dense layers | 18 (odd indices 1,3,...,35) | Anchors during router training |
| Total params | ~28B | But only ~13B active per token |
| Router | Noisy top-k | Gaussian noise prevents expert collapse |

## Setup

```bash
cd code-model-finetuning/sparse_upcycling
pip install -e .
```

## Step 1: Convert Dense → MoE

This clones the FFN in each even-indexed layer into 4 experts and adds a randomly-initialized router. No training happens yet - just model surgery.

```bash
python scripts/convert_dense_to_moe.py --config configs/qwen3_8b_4experts.yaml
```

**What it does internally:**
1. Loads Qwen3-8B in BF16 (~16GB RAM)
2. For layers 0, 2, 4, ..., 34:
   - `layer.mlp` → `SparseMoELayer(4 copies of layer.mlp + router)`
3. Saves the ~28B model to `out/qwen3-8b-moe-4e/`

**Peak memory:** ~56GB CPU RAM (the expanded model in BF16)

## Step 2: Train (continued pre-training)

Full-parameter training of expert FFNs + routers on nvidia/OpenCodeReasoning.
Launched with DeepSpeed ZeRO-3 across 2 GPUs:

```bash
deepspeed --num_gpus 2 scripts/train.py --config configs/qwen3_8b_4experts.yaml
```

**Memory strategy for 2×H200 (282GB VRAM):**

DeepSpeed ZeRO-3 shards weights, gradients, and optimizer states across both GPUs.

```
Per-GPU breakdown:
┌─────────────────────┬────────────────┬──────────────┐
│ Component           │ Precision      │ Per GPU      │
├─────────────────────┼────────────────┼──────────────┤
│ Model weights       │ BF16           │ ~28 GB       │  (56/2 sharded)
│ Optimizer states    │ FP32           │ ~84 GB       │  (168/2 sharded)
│ Gradients           │ BF16           │ ~19 GB       │  (38/2 sharded)
│ Activations         │ BF16 (ckpt)    │ ~10 GB       │
├─────────────────────┼────────────────┼──────────────┤
│ Total per GPU       │                │ ~141 GB ✓    │
└─────────────────────┴────────────────┴──────────────┘
```

**What's trainable (full-parameter, no LoRA):**
- Expert FFNs - full weight updates, max divergence between experts (~19.4B)
- Routers - randomly initialized, 5× higher LR (~0.3M)
- LayerNorm/RMSNorm - adapts to new output distributions (~0.3M)

**What's frozen:**
- Attention layers (Q, K, V, O) - already pretrained, no need to change
- Embeddings - token representations are stable
- Dense anchor FFNs (odd layers) - keep model coherent

## The Training Loss

Two components:

```
total_loss = LM_loss + 0.01 × load_balance_loss
```

1. **LM loss:** Standard next-token prediction (cross-entropy). Computed only on the assistant turn (code solution), not on the problem statement.

2. **Load-balance loss:** Prevents router collapse. Without it, the router sends all tokens to 1-2 experts while others waste compute. Formula: `N × Σ(fraction_routed_i × avg_gate_prob_i)`. Pushes toward uniform expert usage.

## Project Structure

```
sparse_upcycling/
├── configs/
│   └── qwen3_8b_4experts.yaml   # all hyperparameters
├── sparse_upcycling/
│   ├── moe_layer.py              # SparseMoELayer + NoisyTopKRouter
│   └── data.py                   # OpenCodeReasoning dataset wrapper
├── scripts/
│   ├── convert_dense_to_moe.py   # Step 1: model surgery
│   └── train.py                  # Step 2: continued pre-training
└── pyproject.toml
```
