from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple
import json


@dataclass
class X3Config:
    """X3 additions; all Qwen2 backbone dimensions come from checkpoint config."""
    base_model_name_or_path: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    num_experts: int = 4
    expert_names: Tuple[str, ...] = ("coding", "reasoning", "agent", "multilingual")
    adapter_bottleneck_size: int = 32
    router_temperature: float = 1.0
    adapter_scale: float = 1.0
    load_balance_weight: float = 0.01
    entropy_weight: float = 0.0
    inject_after_each_decoder_layer: bool = True

    def __post_init__(self):
        if self.num_experts != len(self.expert_names):
            raise ValueError("num_experts must equal len(expert_names)")
        if self.adapter_bottleneck_size <= 0:
            raise ValueError("adapter_bottleneck_size must be positive")
        if self.router_temperature <= 0:
            raise ValueError("router_temperature must be positive")
        if self.load_balance_weight < 0 or self.entropy_weight < 0:
            raise ValueError("router loss weights must be non-negative")

    @classmethod
    def from_json(cls, path: str) -> "X3Config":
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = json.load(f)
        allowed = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        if "expert_names" in allowed:
            allowed["expert_names"] = tuple(allowed["expert_names"])
        return cls(**allowed)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["expert_names"] = list(self.expert_names)
        return data
