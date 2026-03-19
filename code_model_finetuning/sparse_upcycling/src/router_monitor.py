"""
RouterMonitor - track expert utilization and detect collapse during training.

WHY THIS MATTERS
----------------
MoE models have a failure mode called "expert collapse": the router learns
to send almost all tokens to 1-2 experts while the rest go unused. Those
unused experts waste parameters and the model behaves like a much smaller
dense model.

This is surprisingly common, especially early in training before the
load-balance loss kicks in. Without monitoring you won't notice until
you evaluate the model and wonder why it underperforms.

WHAT COLLAPSE LOOKS LIKE
--------------------------
Fair share for 4 experts = 25% each. Warning signs:

  HEALTHY:        [24.1%  25.3%  26.2%  24.4%]  ← close to 25% each
  EARLY WARNING:  [38.2%  21.0%  19.4%  21.4%]  ← one expert dominant
  COLLAPSING:     [71.3%   9.2%  11.1%   8.4%]  ← one expert takes most tokens
  DEAD EXPERT:    [45.1%  43.2%  11.7%   0.0%]  ← expert 3 is dying

HOW TO RESPOND TO ALERTS
--------------------------
  - Dominant expert (>40%): increase balance_loss_coeff (e.g. 0.01 → 0.05)
  - Dying expert (<2%):     same - higher balance loss pushes uniform usage
  - Persists after 500 steps: reduce learning rate or increase router noise

USAGE
------
    monitor = RouterMonitor(num_experts=4, alert_high=0.40, alert_low=0.02)

    for step, batch in enumerate(dataloader):
        outputs = model(**batch)
        monitor.update(model, step)

        if step % 100 == 0:
            print(monitor.report())

        if monitor.is_collapsing():
            logger.warning("Expert collapse detected - check balance_loss_coeff")
"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch.nn as nn

from src.moe_layer import SparseMoELayer

logger = logging.getLogger(__name__)


class RouterMonitor:
    """
    Tracks per-expert utilization across all MoE layers over training.

    After each forward pass, call monitor.update(model, step) to read the
    routing statistics cached in each SparseMoELayer._last_aux.

    Args:
        num_experts:  number of experts per MoE layer
        alert_high:   flag if any expert handles more than this fraction
                      of tokens. Default 0.40 (1.6× fair share for 4 experts)
        alert_low:    flag if any expert handles fewer than this fraction.
                      Default 0.02 - the expert is effectively dying
        window:       number of recent steps to average over for reporting
    """

    def __init__(
        self,
        num_experts: int,
        alert_high: float = 0.40,
        alert_low: float = 0.02,
        window: int = 100,
    ):
        self.num_experts = num_experts
        self.alert_high = alert_high
        self.alert_low = alert_low
        self.window = window
        self._fair_share = 1.0 / num_experts

        # history[layer_idx] = list of dicts, one per step
        self._history: dict[int, list[dict]] = defaultdict(list)
        self._step = 0

    def update(self, model: nn.Module, step: int | None = None) -> None:
        """
        Read routing stats from all SparseMoELayer modules after a forward pass.
        Call once per training step.
        """
        self._step = step if step is not None else self._step + 1

        for layer_idx, module in enumerate(
            m for m in model.modules() if isinstance(m, SparseMoELayer)
        ):
            aux = module._last_aux
            if not aux:
                continue

            entry = {
                "step": self._step,
                "fraction_routed": aux.get("fraction_routed", None),
                "balance_loss": float(aux.get("balance_loss", 0.0)),
            }

            if entry["fraction_routed"] is None:
                continue

            entry["fraction_routed"] = entry["fraction_routed"].cpu().tolist()

            history = self._history[layer_idx]
            history.append(entry)
            # Keep rolling window only
            if len(history) > self.window:
                self._history[layer_idx] = history[-self.window:]

    def _avg_utilization(self, layer_idx: int) -> list[float]:
        """Average fraction_routed per expert over the recent window."""
        history = self._history.get(layer_idx, [])
        if not history:
            return [self._fair_share] * self.num_experts

        totals = [0.0] * self.num_experts
        for entry in history:
            for i, frac in enumerate(entry["fraction_routed"]):
                if i < self.num_experts:
                    totals[i] += frac
        n = len(history)
        return [t / n for t in totals]

    def is_collapsing(self) -> bool:
        """
        Returns True if ANY MoE layer shows expert collapse signs:
          - any expert above alert_high, OR
          - any expert below alert_low
        """
        for layer_idx in self._history:
            for util in self._avg_utilization(layer_idx):
                if util > self.alert_high or util < self.alert_low:
                    return True
        return False

    def get_alerts(self) -> list[str]:

        alerts = []
        for layer_idx in sorted(self._history):
            for expert_id, util in enumerate(self._avg_utilization(layer_idx)):
                if util > self.alert_high:
                    alerts.append(
                        f"Layer {layer_idx} Expert {expert_id}: {util:.1%} "
                        f"(>{self.alert_high:.0%}) - DOMINANT "
                        f"(fair share={self._fair_share:.1%})"
                    )
                elif util < self.alert_low:
                    alerts.append(
                        f"Layer {layer_idx} Expert {expert_id}: {util:.1%} "
                        f"(<{self.alert_low:.0%}) - DYING "
                        f"(fair share={self._fair_share:.1%})"
                    )
        return alerts

    def report(self) -> str:
        """
        Utilization summary for all MoE layers.

        Example:
            Step 200 | 18 MoE layers | window=100
            Layer  0: [24.8% 25.1% 25.3% 24.8%] bal=1.00  OK
            Layer  2: [38.2% 21.0% 19.4% 21.4%] bal=1.14  ⚠ DOMINANT: expert 0
            Layer  4: [25.0% 25.0% 25.0% 25.0%] bal=1.00  OK
            ...
            Alerts: 1  (increase balance_loss_coeff if persists)
        """
        lines = [
            f"Step {self._step} | {len(self._history)} MoE layers "
            f"| window={self.window}"
        ]

        for layer_idx in sorted(self._history):
            utils = self._avg_utilization(layer_idx)
            history = self._history[layer_idx]
            avg_bal = sum(e["balance_loss"] for e in history) / max(len(history), 1)

            util_str = " ".join(f"{u:.1%}" for u in utils)
            line = f"  Layer {layer_idx:2d}: [{util_str}] bal={avg_bal:.2f}"

            # Annotate problematic experts inline
            flags = []
            for i, u in enumerate(utils):
                if u > self.alert_high:
                    flags.append(f"⚠ DOMINANT: expert {i}")
                elif u < self.alert_low:
                    flags.append(f"⚠ DYING: expert {i}")
            if flags:
                line += "  " + ", ".join(flags)
            else:
                line += "  OK"

            lines.append(line)

        alerts = self.get_alerts()
        if alerts:
            lines.append(
                f"\n  {len(alerts)} alert(s) - consider increasing "
                f"balance_loss_coeff if this persists past 500 steps"
            )
        return "\n".join(lines)
