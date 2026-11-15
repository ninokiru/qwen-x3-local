from __future__ import annotations

from typing import Dict, Optional
import torch
from torch import nn


class ResidualExpert(nn.Module):
    """Small residual bottleneck expert; zero output init preserves the base model."""

    def __init__(self, hidden_size: int, bottleneck_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_size, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(self.norm(x))))


class RoutedAdapterBank(nn.Module):
    """Token-wise soft router over the four trainable X3 expert branches."""

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int,
        expert_names,
        temperature: float = 1.0,
        scale: float = 1.0,
        load_balance_weight: float = 0.01,
        entropy_weight: float = 0.0,
    ):
        super().__init__()
        self.expert_names = tuple(expert_names)
        if not self.expert_names:
            raise ValueError("At least one expert is required")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.scale = float(scale)
        self.load_balance_weight = float(load_balance_weight)
        self.entropy_weight = float(entropy_weight)
        self.router_norm = nn.LayerNorm(hidden_size)
        self.router = nn.Linear(hidden_size, len(self.expert_names), bias=True)
        self.experts = nn.ModuleList(
            [ResidualExpert(hidden_size, bottleneck_size) for _ in self.expert_names]
        )
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.last_metrics: Dict[str, torch.Tensor] = {}
        self._aux_balance = None
        self._aux_entropy = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Shape: [batch, sequence, hidden]. Avoid materializing all experts in
        # one stacked tensor on the memory-constrained CPU path.
        router_logits = self.router(self.router_norm(hidden_states)) / self.temperature
        probabilities = torch.softmax(router_logits.float(), dim=-1).to(hidden_states.dtype)
        outputs = None
        for index, expert in enumerate(self.experts):
            contribution = expert(hidden_states) * probabilities[..., index:index + 1]
            outputs = contribution if outputs is None else outputs + contribution

        flat_probs = probabilities.float().reshape(-1, len(self.expert_names))
        mean_prob = flat_probs.mean(dim=0)
        entropy = -(flat_probs.clamp_min(1e-9).log() * flat_probs).sum(dim=-1).mean()
        # Soft utilization is measurable even before expert-specialized training.
        utilization = mean_prob
        balance = len(self.expert_names) * mean_prob.square().sum() - 1.0
        # Uniform routing gives zero penalty; complete collapse approaches E-1.
        self._aux_balance = balance
        self._aux_entropy = entropy
        self.last_metrics = {
            "router_entropy": entropy.detach(),
            "expert_utilization": utilization.detach(),
            "expert_collapse": (utilization.max() - utilization.min()).detach(),
            "load_balance_loss": balance.detach(),
        }
        return hidden_states + self.scale * outputs

    def auxiliary_loss(self) -> torch.Tensor:
        if self._aux_balance is None or self._aux_entropy is None:
            return next(self.parameters()).new_zeros(())
        balance = self._aux_balance
        entropy = self._aux_entropy
        # Maximize entropy (negative entropy term) while balancing average load.
        return self.load_balance_weight * balance - self.entropy_weight * entropy
