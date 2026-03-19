"""
convert_dense_to_moe.py - Convert a dense Qwen3-8B into a MoE checkpoint.

WHAT THIS SCRIPT DOES
----------------------
1. Load the dense model in BF16 on CPU
2. For each even-indexed layer (generic accessor - works on any HF model):
   a. Extract the MLP via _get_mlp()
   b. Clone it into N Expert objects
   c. Apply Drop-Upcycling: partially re-init each expert with a unique mask
   d. Apply weight scaling: scale down_proj by 1/top_k
   e. Build a SparseMoELayer and replace the MLP via _set_mlp()
3. Apply freeze policy: freeze attention, embeddings, dense anchor FFNs
4. Save the MoE checkpoint + moe_config.json metadata

AFTER CONVERSION
----------------
The model can't generate useful text yet - the routers are near-random.
Run train.py to continue pre-training and teach the routers to route.
Expert weights are already meaningful (copied + partially re-initialized)
so training converges much faster than from scratch.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.moe_layer import Expert, MoEConfig, SparseMoELayer, apply_drop_reinit, apply_weight_scaling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Generic model accessors (improvement 6)
# ------------------------------------------------------------------------------
# These work across model families - Qwen2/3, LLaMA, Mistral, etc.
# No hard-coded "model.model.layers[i].mlp" - we probe the object instead.

def _get_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """
    Find the main list of transformer decoder layers in any HF model.

    Tries common attribute paths in order:
      model.model.layers   (Qwen2, LLaMA, Mistral, Phi-3)
      model.layers         (some older models)
      Falls back to scanning for the longest ModuleList in the model.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    # Generic fallback: find longest ModuleList (heuristic - the layer stack)
    best: list[nn.Module] = []
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) > len(best):
            best = list(module)
    if best:
        return best
    raise ValueError(
        "Could not find transformer layers. Expected model.model.layers "
        "or model.layers - check your model's architecture."
    )


def _get_mlp(layer: nn.Module) -> nn.Module | None:
    """Extract the FFN/MLP sub-module from a transformer layer."""
    if hasattr(layer, "mlp"):
        return layer.mlp
    if hasattr(layer, "feed_forward"):
        return layer.feed_forward
    return None


def _set_mlp(layer: nn.Module, new_mlp: nn.Module) -> None:
    """Replace the FFN/MLP in a transformer layer."""
    if hasattr(layer, "mlp"):
        layer.mlp = new_mlp
    elif hasattr(layer, "feed_forward"):
        layer.feed_forward = new_mlp
    else:
        raise AttributeError(
            f"Cannot find MLP attribute on {type(layer).__name__}. "
            "Expected 'mlp' or 'feed_forward'."
        )


# ------------------------------------------------------------------------------
# Freeze policy
# ------------------------------------------------------------------------------

def apply_freeze_policy(model: nn.Module) -> tuple[int, int]:
    """
    Freeze everything except MoE experts, routers, and LayerNorms.

    What we freeze and why:
      Embeddings    - token lookup table, already well-trained, changing it
                      destabilizes everything built on top
      Attention     - Q/K/V/O matrices, trained on trillions of tokens,
                      MoE conversion only changes FFNs not token relationships
      Dense FFNs    - odd-indexed anchor layers stay frozen to prevent drift

    What we train:
      Expert FFNs   - must differentiate, need full-param updates
      Routers       - randomly initialized, must learn from scratch
      LayerNorms    - must adapt to the new MoE output distributions
                      (small param count, low risk)

    Returns:
        (trainable_count, frozen_count) parameter counts
    """
    # Step 1: freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: unfreeze MoE layers (experts + routers)
    for module in model.modules():
        if isinstance(module, SparseMoELayer):
            for param in module.parameters():
                param.requires_grad = True

    # Step 3: unfreeze LayerNorm / RMSNorm
    for module in model.modules():
        cls_name = type(module).__name__.lower()
        if "rmsnorm" in cls_name or "layernorm" in cls_name:
            for param in module.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


# ------------------------------------------------------------------------------
# Main conversion
# ------------------------------------------------------------------------------

def convert(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name   = cfg["model"]["name_or_path"]
    output_dir   = Path(cfg["model"]["output_dir"])
    num_experts  = cfg["moe"]["num_experts"]
    top_k        = cfg["moe"]["top_k"]
    strategy     = cfg["moe"]["layer_selection"]
    drop_ratio   = cfg["moe"].get("drop_reinit_ratio", 0.2)
    weight_scale = cfg["moe"].get("weight_scale") or (1.0 / top_k)
    seed         = cfg["moe"].get("seed", 42)

    logger.info("=" * 60)
    logger.info("Dense -> MoE Upcycling")
    logger.info("  Model:          %s", model_name)
    logger.info("  Experts:        %d  (top-%d routing)", num_experts, top_k)
    logger.info("  Drop-reinit:    %.0f%%  (Drop-Upcycling)", drop_ratio * 100)
    logger.info("  Weight scale:   %.3f  (1/top_k)", weight_scale)
    logger.info("  Layer strategy: %s", strategy)
    logger.info("=" * 60)

    logger.info("Loading dense model (BF16, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    layers = _get_transformer_layers(model)

    logger.info("Found %d transformer layers, hidden_size=%d", num_layers, hidden_size)

    # Select which layers become MoE
    if strategy == "alternating":
        moe_indices = list(range(0, num_layers, 2))
    elif strategy == "all":
        moe_indices = list(range(num_layers))
    else:
        raise ValueError(f"Unknown layer_selection: {strategy}")

    moe_config = MoEConfig(
        num_experts=num_experts,
        top_k=top_k,
        balance_loss_coeff=cfg["moe"].get("load_balance_coeff", 0.01),
        drop_reinit_ratio=drop_ratio,
        router_noise_std=cfg["moe"].get("router_noise_std", 0.1),
        weight_scale=weight_scale,
    )

    logger.info("Converting %d/%d layers to MoE...", len(moe_indices), num_layers)

    for layer_idx in moe_indices:
        layer = layers[layer_idx]
        mlp = _get_mlp(layer)

        if mlp is None:
            logger.warning("Layer %d: no MLP found, skipping", layer_idx)
            continue

        param_count = sum(p.numel() for p in mlp.parameters()) / 1e6

        # Build experts: clone -> drop-reinit -> weight scale
        experts = nn.ModuleList()
        layer_seed = seed + layer_idx * 1000  # unique per layer
        for expert_id in range(num_experts):
            expert = Expert.from_mlp(mlp)
            apply_drop_reinit(expert, drop_ratio, expert_id, layer_seed)
            apply_weight_scaling(expert, weight_scale)
            experts.append(expert)

        moe_layer = SparseMoELayer(
            hidden_size=hidden_size,
            experts=experts,
            config=moe_config,
        )
        _set_mlp(layer, moe_layer)

        logger.info(
            "  Layer %2d -> MoE  (%d experts × %.0fM params, drop=%.0f%%, scale=%.2f)",
            layer_idx, num_experts, param_count, drop_ratio * 100, weight_scale,
        )

    # Apply freeze policy
    trainable, frozen = apply_freeze_policy(model)
    total = trainable + frozen

    logger.info("\nConversion complete:")
    logger.info("  Total params:     %.2fB", total / 1e9)
    logger.info("  Trainable params: %.2fB  (%.1f%%)", trainable / 1e9, 100 * trainable / total)
    logger.info("  Frozen params:    %.2fB  (%.1f%%)", frozen / 1e9, 100 * frozen / total)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("\nSaving to %s ...", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    moe_meta = {
        "base_model": model_name,
        "num_experts": num_experts,
        "top_k": top_k,
        "moe_layer_indices": moe_indices,
        "layer_selection": strategy,
        "drop_reinit_ratio": drop_ratio,
        "weight_scale": weight_scale,
        "total_params_B": round(total / 1e9, 2),
        "trainable_params_B": round(trainable / 1e9, 2),
    }
    with open(output_dir / "moe_config.json", "w") as f:
        json.dump(moe_meta, f, indent=2)

    logger.info("Done. Next: python scripts/train.py --config %s", config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    convert(args.config)
