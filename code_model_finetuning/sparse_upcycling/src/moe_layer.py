"""
Mixture-of-Experts layer for sparse upcycling.

WHAT IS SPARSE UPCYCLING?
-------------------------
A dense transformer has one FFN (feed-forward network) per layer.
Sparse upcycling duplicates that FFN into N "experts" and adds a tiny
router network that decides which experts process each token.

For Qwen3-8B with 4 experts and top-2 routing:
  - Each token only activates 2 out of 4 experts → same FLOPs as ~2× the FFN
  - But the model has 4× the FFN parameters → more capacity to specialize
  - Total params go from ~8B to ~28B, but active params per token ≈ 13B

WHY ALTERNATING LAYERS?
----------------------
Converting ALL layers to MoE means every layer has a randomly-initialized
router. That's too much instability at once. Alternating (even=MoE, odd=dense)
keeps half the layers as stable "anchors" while routers learn to route.

DROP-UPCYCLING (Nakamura et al., ICLR 2025)
--------------------------------------------
Problem: All experts start as identical copies → router has no gradient
signal to prefer one expert over another → differentiation is slow.

Solution: Randomly re-initialize a fraction (20%) of each expert's weights
with a DIFFERENT random mask per expert. This introduces instant diversity
while preserving most of the pretrained knowledge.

  Expert 0: mask_0 → 20% re-init at positions [3, 17, 42, ...]
  Expert 1: mask_1 → 20% re-init at positions [1, 8,  29, ...]  (different mask)
  Expert 2: mask_2 → 20% re-init at positions [5, 12, 38, ...]  (different mask)
  ...

WEIGHT SCALING
--------------
After upcycling, each token's output is a weighted sum of top-k expert
outputs (weights sum to 1.0). With top-2 routing, the effective output
magnitude is roughly the same as the original single FFN - but only if
we scale the experts' output projections by 1/top_k at init.

Without scaling: output ≈ 2× original magnitude → large loss spike at
the start of training as LayerNorm compensates.
With scaling (down_proj × 0.5 for top-2): output ≈ original magnitude
→ smooth loss curve from step 1.

NEAR-UNIFORM ROUTER INIT
-------------------------
The router's gate weights are initialized very small (× 0.01 of Kaiming).
This makes initial logits near-zero → softmax is near-uniform → all
experts get roughly equal probability at the start. Training then
gradually sharpens the routing as experts differentiate.

Without this: random large logits → one expert gets all tokens from step 1
→ training collapses immediately.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class MoEConfig:
    """All hyperparameters for a MoE layer in one place."""

    num_experts: int = 4
    top_k: int = 2

    # Load-balancing loss coefficient. Added to LM loss:
    #   total_loss = lm_loss + balance_loss_coeff * balance_loss
    # 0.01 is the Switch Transformer default. Too high → equal load but poor
    # specialization. Too low → routing collapses to 1-2 experts.
    balance_loss_coeff: float = 0.01

    # Drop-Upcycling: fraction of expert weights to randomly re-initialize.
    # 0.0 = vanilla copy (all experts start identical, slow differentiation)
    # 0.2 = re-init 20% per expert (ICLR 2025 recommended value)
    # 0.5 = aggressive - more diversity but loses more pretrained knowledge
    drop_reinit_ratio: float = 0.2

    # Router noise std during training (encourages exploration).
    router_noise_std: float = 0.1

    # Output scaling per expert. None = auto (1/top_k).
    # Scales down_proj so output magnitude matches original dense FFN.
    weight_scale: float | None = None


# -----------------------------------------------------------------------------
# Expert (improvement 4: separate class instead of raw nn.Module copy)
# -----------------------------------------------------------------------------

class Expert(nn.Module):
    """
    A single expert FFN - a SwiGLU MLP matching Qwen3's architecture:
        output = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    Each expert is initialized as a copy of the original dense MLP, then
    optionally partially re-initialized via Drop-Upcycling to break symmetry.

    Having Expert as a separate class (vs. raw copy of mlp module) means:
    - We can identify experts by type in the RouterMonitor and freeze logic
    - We can add expert-specific methods later (e.g., per-expert adapter)
    - The code is explicit about what "an expert" is
    """

    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: element-wise multiply gated and up projections, then project down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    @classmethod
    def from_mlp(cls, mlp: nn.Module) -> "Expert":
        """
        Create an Expert from a HuggingFace Qwen MLP module.
        Deep-copies the weights so the original is not mutated.
        """
        return cls(
            gate_proj=copy.deepcopy(mlp.gate_proj),
            up_proj=copy.deepcopy(mlp.up_proj),
            down_proj=copy.deepcopy(mlp.down_proj),
        )


# -----------------------------------------------------------------------------
# Drop-Upcycling helpers (improvement 1)
# -----------------------------------------------------------------------------

def apply_drop_reinit(expert: Expert, ratio: float, expert_id: int, layer_seed: int) -> Expert:
    """
    Partially re-initialize an expert's weights (Drop-Upcycling).

    Each expert gets a unique seed (layer_seed + expert_id) so they get
    different masks - instant diversity from initialization.

    Only weight matrices are re-initialized (not biases), and only 2D+
    tensors (linear layer weights), not scalars like LayerNorm scales.

    Args:
        expert:      the Expert to modify in-place
        ratio:       fraction of each weight tensor to re-initialize (0.0–1.0)
        expert_id:   index of this expert (0, 1, 2, ...) - affects random seed
        layer_seed:  base seed for this layer (different per transformer layer)
    """
    if ratio <= 0.0:
        return expert

    rng = torch.Generator()
    rng.manual_seed(layer_seed + expert_id)

    for param in expert.parameters():
        if param.dim() < 2:
            continue  # skip biases and scalars

        fan_in = param.shape[1]
        bound = math.sqrt(3.0 / fan_in) if fan_in > 0 else 1.0

        # Random binary mask: True where we re-initialize
        mask = torch.rand(param.shape, generator=rng) < ratio
        fresh = torch.empty_like(param).uniform_(-bound, bound)
        param.data = torch.where(mask, fresh, param.data)

    return expert


# -----------------------------------------------------------------------------
# Weight scaling helper (improvement 2)
# -----------------------------------------------------------------------------

def apply_weight_scaling(expert: Expert, scale: float) -> Expert:
    """
    Scale the expert's down_proj to preserve output distribution after upcycling.

    WHY down_proj?
    It's the output projection of the FFN. Scaling it scales the expert's
    entire contribution. For top-2 routing with weights that sum to 1.0,
    scaling by 0.5 ensures the expected sum of expert contributions matches
    the original single-FFN output magnitude.

    Applied once at initialization - not during forward passes.
    """
    if abs(scale - 1.0) < 1e-6:
        return expert
    with torch.no_grad():
        expert.down_proj.weight.mul_(scale)
        if expert.down_proj.bias is not None:
            expert.down_proj.bias.mul_(scale)
    return expert


# -----------------------------------------------------------------------------
# Router (improvement 5: near-uniform init)
# -----------------------------------------------------------------------------

class TopKRouter(nn.Module):
    """
    Routes each token to the top-k scoring experts.

    Key design decisions:
    1. Near-uniform init (gate.weight × 0.01): initial logits are near-zero
       → softmax is near-uniform → no expert dominates before training starts.
    2. Fixed Gaussian noise during training (not learned): simpler than a
       learned noise projection, same exploration benefit.
    3. Softmax over top-k only (not all experts): the selected experts'
       weights sum to 1.0, which combines cleanly with weight scaling.
    4. Full softmax over all experts for the load-balance loss: we need
       probabilities across all experts to measure utilization imbalance.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, noise_std: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Near-uniform init: scale down Kaiming init by 100×
        # → initial logits are tiny → near-uniform routing from step 1
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.gate.weight.mul_(0.01)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: (num_tokens, hidden_size)

        Returns:
            top_indices: (num_tokens, top_k) - which experts each token goes to
            top_weights: (num_tokens, top_k) - softmax weights for chosen experts
            aux: dict with 'balance_loss', 'fraction_routed', 'router_probs'
                 (stored for RouterMonitor and loss collection)
        """
        logits = self.gate(x)  # (num_tokens, num_experts)

        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        # Full softmax for load-balance loss (needs all-expert probabilities)
        router_probs = F.softmax(logits, dim=-1)  # (num_tokens, num_experts)

        # Top-k selection
        top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
        top_weights = F.softmax(top_values, dim=-1)  # sum=1.0 over top-k per token

        # -- Load-balancing loss --------------------------------
        # fraction_routed[i] = fraction of tokens that chose expert i (in top-k)
        # avg_gate_prob[i]   = mean router probability for expert i
        # L_balance = N × Σ(fraction_routed_i × avg_gate_prob_i)
        # Perfect balance → L_balance = 1.0
        # All tokens → expert 0 → L_balance = N
        num_tokens = x.shape[0]
        expert_mask = torch.zeros(num_tokens, self.num_experts, device=x.device, dtype=x.dtype)
        expert_mask.scatter_add_(
            1, top_indices, torch.ones_like(top_indices, dtype=x.dtype)
        )
        fraction_routed = expert_mask.mean(dim=0)   # (num_experts,)
        avg_gate_prob = router_probs.mean(dim=0)    # (num_experts,)
        balance_loss = self.num_experts * (fraction_routed * avg_gate_prob).sum()

        aux = {
            "balance_loss": balance_loss,
            "fraction_routed": fraction_routed.detach(),
            "router_probs": router_probs.detach(),
        }

        return top_indices, top_weights, aux


# ----------------------------------------------------------------------------
# MoE layer
# ----------------------------------------------------------------------------

class SparseMoELayer(nn.Module):
    """
    Replaces a single dense FFN with N Expert FFNs + a TopKRouter.

    After conversion all experts start as (partially re-initialized) copies
    of the original FFN. They differentiate during training as the router
    learns to send different token types to different experts.

    The forward() returns a single tensor (same shape as input) so it can
    drop-in replace any HuggingFace MLP module without touching the model's
    decoder layer code. Auxiliary data is stored in self._last_aux and
    collected by collect_balance_loss() after each forward pass.
    """

    def __init__(self, hidden_size: int, experts: nn.ModuleList, config: MoEConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.experts = experts
        self.config = config
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=len(experts),
            top_k=config.top_k,
            noise_std=config.router_noise_std,
        )
        self._last_aux: dict = {}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            output: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat = hidden_states.view(-1, hidden_size)   # (num_tokens, hidden_size)

        top_indices, top_weights, aux = self.router(flat)
        self._last_aux = aux  # RouterMonitor reads this after each forward

        output = torch.zeros_like(flat)

        # Dispatch: for each top-k slot, find tokens routed there and run expert
        for k in range(self.config.top_k):
            expert_indices = top_indices[:, k]   # (num_tokens,) - which expert for slot k
            expert_weights = top_weights[:, k]   # (num_tokens,) - weight for slot k

            for expert_id in range(len(self.experts)):
                mask = expert_indices == expert_id
                if not mask.any():
                    continue

                token_subset = flat[mask]                              # (n, hidden_size)
                expert_out = self.experts[expert_id](token_subset)     # (n, hidden_size)
                weight = expert_weights[mask].unsqueeze(-1)            # (n, 1)
                output[mask] += weight * expert_out

        return output.view(batch_size, seq_len, hidden_size)


# -----------------------------------------------------------------------------
# Loss collection utility
# -----------------------------------------------------------------------------

def collect_balance_loss(model: nn.Module, coeff: float = 0.01) -> torch.Tensor:
    """
    Walk all SparseMoELayer modules and sum their load-balancing losses.

    Call after every forward pass, before backward:
        total_loss = lm_loss + collect_balance_loss(model, coeff=0.01)
        total_loss.backward()

    Averages across all MoE layers so the loss scale doesn't grow with
    the number of MoE layers.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0
    for module in model.modules():
        if isinstance(module, SparseMoELayer) and module._last_aux:
            total = total + module._last_aux.get("balance_loss", 0.0)
            count += 1
    if count > 0:
        total = total / count
    return coeff * total
