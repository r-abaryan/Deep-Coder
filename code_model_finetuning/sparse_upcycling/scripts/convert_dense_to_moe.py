"""
convert_dense_to_moe.py — The core upcycling step.

Takes a dense Qwen3-8B checkpoint and produces a MoE checkpoint where
selected layers have their FFN replaced with SparseMoELayer.

WHAT THIS SCRIPT DOES (step by step):
─────────────────────────────────────
1. Load the dense Qwen3-8B model (in FP16/BF16 to fit in RAM)
2. For each layer designated as MoE (alternating = even indices):
   a. Extract the existing MLP module (gate_proj, up_proj, down_proj)
   b. Create a SparseMoELayer with 4 experts, each a clone of that MLP
   c. Create a randomly-initialized router
   d. Replace the original MLP with the SparseMoELayer
3. Save the full MoE model + a modified config

After conversion the model can't be used directly for inference yet —
the routers are random and will output garbage. You need the training
step to teach the routers which experts to use for which tokens.

MEMORY NOTE:
The converted model has ~28B params (4× the FFNs on 18 layers).
In BF16 that's ~56GB — fits in a single H200's VRAM or ~60GB CPU RAM.
We load the dense model first (~16GB), then expand in-place.
Peak memory during conversion: ~56GB CPU RAM.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.moe_layer import SparseMoELayer


def get_moe_layer_indices(num_layers: int, strategy: str) -> list[int]:
    """
    Decide which layers become MoE.

    "alternating": even indices (0, 2, 4, ...) become MoE.
    Odd indices stay dense — they act as stable anchors during training.

    For Qwen3-8B with 36 layers → 18 MoE layers, 18 dense layers.
    """
    if strategy == "alternating":
        return list(range(0, num_layers, 2))
    elif strategy == "all":
        return list(range(num_layers))
    else:
        raise ValueError(f"Unknown layer_selection strategy: {strategy}")


def convert(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name_or_path"]
    output_dir = Path(cfg["model"]["output_dir"])
    num_experts = cfg["moe"]["num_experts"]
    top_k = cfg["moe"]["top_k"]
    layer_strategy = cfg["moe"]["layer_selection"]

    print(f"Loading dense model: {model_name}")
    # Load in BF16 to save memory. CPU only - we're just doing surgery.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    moe_indices = get_moe_layer_indices(num_layers, layer_strategy)
    print(f"Model has {num_layers} layers, hidden_size={hidden_size}")
    print(f"Converting {len(moe_indices)} layers to MoE: {moe_indices}")
    print(f"  {num_experts} experts per layer, top-{top_k} routing")

    # ------------------------------- Layer surgery ----------------------------------
    # Qwen3 layers are at: model.model.layers[i].mlp
    for idx in moe_indices:
        layer = model.model.layers[idx]
        original_mlp = layer.mlp

        print(f"  Layer {idx}: replacing MLP with SparseMoELayer "
              f"({num_experts} experts, {_count_params(original_mlp):.1f}M params each)")

        moe = SparseMoELayer(
            hidden_size=hidden_size,
            expert_module=original_mlp,
            num_experts=num_experts,
            top_k=top_k,
        )

        # Replace the MLP. The original MLP is now garbage-collected
        # (except for the copies living inside moe.experts).
        layer.mlp = moe

    total_params = sum(p.numel() for p in model.parameters())
    moe_params = sum(
        p.numel() for idx in moe_indices
        for p in model.model.layers[idx].mlp.parameters()
    )
    print(f"\nConversion complete:")
    print(f"  Total params: {total_params / 1e9:.1f}B")
    print(f"  MoE params:   {moe_params / 1e9:.1f}B")
    print(f"  Dense params:  {(total_params - moe_params) / 1e9:.1f}B")
    print(f"  Active params/token: ~{(total_params - moe_params + moe_params / num_experts * top_k) / 1e9:.1f}B")

    # ── Save ─────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the model weights
    print(f"\nSaving MoE model to {output_dir} ...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save MoE metadata so the training script knows which layers are MoE
    moe_meta = {
        "base_model": model_name,
        "num_experts": num_experts,
        "top_k": top_k,
        "moe_layer_indices": moe_indices,
        "layer_selection": layer_strategy,
        "total_params_B": round(total_params / 1e9, 1),
    }
    with open(output_dir / "moe_config.json", "w") as f:
        json.dump(moe_meta, f, indent=2)

    print(f"Done. MoE checkpoint saved to: {output_dir}")
    print(f"Next step: python scripts/train.py --config {config_path}")


def _count_params(module: torch.nn.Module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dense model to MoE")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    convert(args.config)
