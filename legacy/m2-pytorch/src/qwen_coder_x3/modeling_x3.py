from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from .configuration_x3 import X3Config
from .adapters import RoutedAdapterBank
from .decoder_layer import X3DecoderLayer


@dataclass
class X3Output:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]
    aux_router_loss: torch.Tensor
    router_metrics: Dict[str, torch.Tensor]
    past_key_values: Any = None
    hidden_states: Any = None
    attentions: Any = None
    raw_output: Any = None

    @property
    def total_loss(self):
        if self.loss is None:
            return self.aux_router_loss
        return self.loss + self.aux_router_loss


class X3ForCausalLM(nn.Module):
    """Qwen2.5-Coder backbone with explicit post-block routed adapters.

    The Qwen2 decoder block itself is retained as a child module, so its
    attention, RoPE, GQA, RMSNorm, SwiGLU, and cache implementation are not
    re-created or approximated here.
    """

    def __init__(self, backbone: nn.Module, x3_config: Optional[X3Config] = None):
        super().__init__()
        self.backbone = backbone
        self.x3_config = x3_config or X3Config()
        base_cfg = backbone.config
        self._validate_base_config(base_cfg)
        decoder = backbone.model.layers
        wrapped = []
        for layer in decoder:
            ref = next(layer.parameters())
            adapter = RoutedAdapterBank(
                hidden_size=int(base_cfg.hidden_size),
                bottleneck_size=self.x3_config.adapter_bottleneck_size,
                expert_names=self.x3_config.expert_names,
                temperature=self.x3_config.router_temperature,
                scale=self.x3_config.adapter_scale,
                load_balance_weight=self.x3_config.load_balance_weight,
                entropy_weight=self.x3_config.entropy_weight,
            ).to(device=ref.device, dtype=ref.dtype)
            wrapped.append(X3DecoderLayer(layer, adapter))
        # Preserve the original ModuleList interface expected by Qwen2Model.
        backbone.model.layers = nn.ModuleList(wrapped)

    @staticmethod
    def _validate_base_config(cfg):
        required = ("hidden_size", "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "vocab_size", "intermediate_size")
        missing = [name for name in required if not hasattr(cfg, name)]
        if missing:
            raise ValueError(f"Base config missing required Qwen2 fields: {missing}")
        if int(cfg.num_attention_heads) % int(cfg.num_key_value_heads):
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    @classmethod
    def from_pretrained_base(
        cls,
        model_name_or_path: str,
        *,
        x3_config: Optional[X3Config] = None,
        torch_dtype=None,
        device_map=None,
        trust_remote_code: bool = False,
        low_cpu_mem_usage: bool = True,
    ):
        kwargs = {"trust_remote_code": trust_remote_code,
                  "low_cpu_mem_usage": low_cpu_mem_usage}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            kwargs["device_map"] = device_map
        from transformers import AutoModelForCausalLM
        backbone = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        return cls(backbone=backbone, x3_config=x3_config)

    def forward(self, *args, **kwargs) -> X3Output:
        # The backbone receives all arguments unchanged, including masks,
        # position IDs, use_cache and past_key_values. Unsupported cache APIs
        # therefore fail in Transformers itself rather than being discarded.
        output = self.backbone(*args, **kwargs)
        adapters = [layer.adapter for layer in self.backbone.model.layers]
        output_device = output.logits.device
        aux_losses = [adapter.auxiliary_loss().to(output_device) for adapter in adapters]
        aux = torch.stack(aux_losses).mean() if aux_losses else output.logits.new_zeros(())
        metrics: Dict[str, torch.Tensor] = {}
        if adapters and adapters[0].last_metrics:
            for key in ("router_entropy", "expert_utilization", "expert_collapse"):
                values = [adapter.last_metrics[key].to(output_device) for adapter in adapters]
                metrics[key] = torch.stack(values).mean(dim=0)
        return X3Output(
            logits=output.logits,
            loss=getattr(output, "loss", None),
            aux_router_loss=aux,
            router_metrics=metrics,
            past_key_values=getattr(output, "past_key_values", None),
            hidden_states=getattr(output, "hidden_states", None),
            attentions=getattr(output, "attentions", None),
            raw_output=output,
        )

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)

    @property
    def config(self):
        return self.backbone.config

    def save_x3_pretrained(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        torch.save({
            "format": "qwen-coder-x3-m2",
            "x3_config": self.x3_config.to_dict(),
            "state_dict": self.state_dict(),
            "base_model": self.x3_config.base_model_name_or_path,
        }, os.path.join(output_dir, "x3_m2.pt"))
        save_config = getattr(self.backbone.config, "save_pretrained", None)
        if callable(save_config):
            save_config(output_dir)
