"""
train.py - Continue pre-training the upcycled MoE model.

WHAT HAPPENS DURING TRAINING
-----------------------------
After conversion, experts are partially re-initialized copies of the original
FFN (Drop-Upcycling gave them instant diversity). Routers are near-random
(near-uniform init). Training does two things:

1. ROUTER LEARNING: Routers learn which tokens to send where. Near-uniform
   init means no expert dominates from step 1 - all experts get meaningful
   gradient signal from the beginning. Routing sharpens over training.

2. EXPERT DIFFERENTIATION: Full-parameter training lets each expert's weights
   diverge freely. Different tokens --> different experts --> different gradients
   --> experts specialize (e.g., one expert for loops, one for I/O, etc.)

WHAT WE FREEZE AND WHY
------------------------
  ┌-----------------┬----------┬--------------------------------------┐
  │ Component       │ Trainable│ Reason                               │
  ├-----------------┼----------┼--------------------------------------┤
  │ Embeddings      │ FROZEN   │ Token representations well-learned.  │
  │ Attention       │ FROZEN   │ Token relationships well-learned.    │
  │ Dense FFNs      │ FROZEN   │ Anchor layers prevent drift.         │
  │ Expert FFNs     │ TRAINED  │ Must differentiate (full-param).     │
  │ Routers         │ TRAINED  │ Randomly initialized, learn routing. │
  │ LayerNorm/RMS   │ TRAINED  │ Adapt to MoE output distributions.   │
  └-----------------┴----------┴--------------------------------------┘

WHY WE RE-APPLY FREEZE HERE (not rely on convert_dense_to_moe.py)
-------------------------------------------------------------------
PyTorch's save_pretrained() saves weight VALUES only - not requires_grad flags.
When we load with from_pretrained(), all parameters default to requires_grad=True.
So the freeze applied in convert_dense_to_moe.py is lost. We must re-apply it
here before building the optimizer, otherwise we'd train ~28B params instead
of ~19.5B, wasting memory and compute on attention weights that don't need updating.

MULTI-GPU WITH ACCELERATE + DEEPSPEED ZeRO-3
---------------------------------------------
Accelerate wraps the training loop to distribute across 2×H200 transparently.
ZeRO-3 shards model weights + gradients + optimizer states across GPUs.

Each GPU holds:
  ~28GB  weights (56GB / 2)
  ~84GB  optimizer states (168GB / 2)
  ~19GB  gradients (38GB / 2)
  ~10GB  activations (grad checkpointing)
  ----------------
  ~141GB per GPU (H200 has 141GB)

Launch command (2 GPUs):
  accelerate launch --num_processes 2 --config_file configs/accelerate_config.yaml \\
      scripts/train.py --config configs/qwen3_8b_4experts.yaml
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.convert_dense_to_moe import apply_freeze_policy
from src.data import OpenCodeReasoningDataset, collate_fn
from src.moe_layer import collect_balance_loss
from src.router_monitor import RouterMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["model"]["output_dir"])
    moe_config_path = output_dir / "moe_config.json"

    if not moe_config_path.exists():
        logger.error("%s not found. Run convert_dense_to_moe.py first.", moe_config_path)
        sys.exit(1)

    with open(moe_config_path) as f:
        moe_meta = json.load(f)

    # ---- Accelerator (handles multi-GPU + DeepSpeed ZeRO-3) ----
    # Reads configs/accelerate_config.yaml automatically when launched via
    # `accelerate launch`. Falls back to single-GPU if launched directly.
    ds_cfg = cfg.get("deepspeed", {})
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        log_with="tensorboard",
        project_dir=str(output_dir / "logs"),
    )

    if accelerator.is_main_process:
        logger.info("Loading MoE model from %s", output_dir)
        logger.info("  %d experts, top-%d, Drop-reinit=%.0f%%",
                    moe_meta["num_experts"], moe_meta["top_k"],
                    moe_meta.get("drop_reinit_ratio", 0.0) * 100)
        logger.info("  Num GPUs: %d", accelerator.num_processes)

    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------- Re-apply freeze policy ---------------------------------
    # CRITICAL: requires_grad flags are NOT saved by save_pretrained().
    # Loading with from_pretrained() defaults all params to requires_grad=True.
    # We must re-apply the freeze here to only train experts+routers+norms.
    trainable, frozen = apply_freeze_policy(model)
    total = trainable + frozen
    if accelerator.is_main_process:
        logger.info("Freeze policy applied:")
        logger.info("  Trainable: %.2fB (%.1f%%) — experts, routers, norms",
                    trainable / 1e9, 100 * trainable / total)
        logger.info("  Frozen:    %.2fB (%.1f%%) — attention, embeddings, dense FFNs",
                    frozen / 1e9, 100 * frozen / total)

    model.gradient_checkpointing_enable()

    # ------------------------ Dataset -----------------------------------------------
    dataset = OpenCodeReasoningDataset(
        tokenizer=tokenizer,
        max_seq_len=cfg["training"]["max_seq_len"],
        split=cfg["data"]["split"],
        max_samples=cfg["data"].get("max_samples"),
    )
    collate = functools.partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["per_device_batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
    )

    # -------------------------- Optimizer ---------------------------------------------
    # Routers get 5× higher LR: they start from near-random init while
    # expert FFNs start from pretrained weights and need gentler updates.
    router_params, expert_params, norm_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "router" in name:
            router_params.append(param)
        elif "norm" in name.lower():
            norm_params.append(param)
        else:
            expert_params.append(param)

    base_lr = cfg["training"]["learning_rate"]
    optimizer = torch.optim.AdamW([
        {"params": expert_params, "lr": base_lr},
        {"params": norm_params,   "lr": base_lr},
        {"params": router_params, "lr": base_lr * 5},
    ], weight_decay=cfg["training"]["weight_decay"])

    total_steps = (
        len(dataloader)
        // cfg["training"]["gradient_accumulation_steps"]
        * cfg["training"]["epochs"]
    )
    scheduler = get_scheduler(
        cfg["training"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * cfg["training"]["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    # -- Accelerate: prepare everything for distributed training --
    # This wraps model, optimizer, dataloader, scheduler so they work
    # seamlessly across multiple GPUs with ZeRO-3 sharding.
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # ---------------------- RouterMonitor (main process only) ---------------------
    monitor = RouterMonitor(
        num_experts=moe_meta["num_experts"],
        alert_high=0.40,
        alert_low=0.02,
        window=100,
    )

    # ----------------------- Training loop -----------------------------------------
    balance_coeff = cfg["moe"]["load_balance_coeff"]
    log_steps     = cfg["training"]["logging_steps"]
    grad_accum    = cfg["training"]["gradient_accumulation_steps"]

    global_step = 0

    for epoch in range(cfg["training"]["epochs"]):
        if accelerator.is_main_process:
            logger.info("=" * 60)
            logger.info("Epoch %d / %d", epoch + 1, cfg["training"]["epochs"])

        model.train()
        pbar = tqdm(dataloader, desc="Training", disable=not accelerator.is_main_process)
        accum_lm = accum_aux = 0.0

        for step, batch in enumerate(pbar):
            # accelerator.accumulate handles gradient accumulation correctly
            # across GPUs and with DeepSpeed ZeRO
            with accelerator.accumulate(model):
                outputs = model(**batch)
                lm_loss = outputs.loss

                # Unwrap to access SparseMoELayer._last_aux for balance loss
                unwrapped = accelerator.unwrap_model(model)
                aux_loss = collect_balance_loss(unwrapped, coeff=balance_coeff)

                loss = lm_loss + aux_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), cfg["training"]["max_grad_norm"]
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            accum_lm  += lm_loss.item()
            accum_aux += aux_loss.item()

            if accelerator.sync_gradients:
                global_step += 1

                if accelerator.is_main_process:
                    monitor.update(unwrapped, step=global_step)

                if global_step % log_steps == 0 and accelerator.is_main_process:
                    avg_lm  = accum_lm  / log_steps
                    avg_aux = accum_aux / log_steps
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix(lm=f"{avg_lm:.4f}", aux=f"{avg_aux:.4f}", lr=f"{lr:.2e}")
                    accum_lm = accum_aux = 0.0

                    if monitor.is_collapsing():
                        logger.warning("Expert collapse detected!!")
                        for alert in monitor.get_alerts():
                            logger.warning("  %s", alert)

                if global_step % (log_steps * 5) == 0 and accelerator.is_main_process:
                    logger.info("\n%s", monitor.report())

                if global_step % cfg["training"]["save_steps"] == 0:
                    accelerator.wait_for_everyone()
                    ckpt = output_dir / f"checkpoint-{global_step}"
                    if accelerator.is_main_process:
                        logger.info("Saving checkpoint to %s", ckpt)
                    accelerator.save_state(str(ckpt))

    accelerator.wait_for_everyone()
    final = output_dir / "final"
    if accelerator.is_main_process:
        logger.info("Saving final model to %s", final)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(final)
        tokenizer.save_pretrained(final)
        logger.info("Training completee.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
