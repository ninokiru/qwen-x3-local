from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import re
import torch


@dataclass
class MappingReport:
    copied: int
    missing_in_x3: List[str]
    unexpected_in_x3: List[str]
    shape_mismatches: List[str]

    @property
    def ok(self) -> bool:
        return not (self.missing_in_x3 or self.unexpected_in_x3 or self.shape_mismatches)

    def to_dict(self):
        return asdict(self) | {"ok": self.ok}


def x3_key_for_base_key(key: str) -> str:
    """Map HF Qwen2ForCausalLM keys into the explicitly wrapped X3 backbone."""
    match = re.match(r"^(model\.layers\.\d+)\.(.+)$", key)
    if match:
        return f"backbone.{match.group(1)}.base_layer.{match.group(2)}"
    return f"backbone.{key}"


def map_base_state_dict(base_state: Dict[str, torch.Tensor],
                        x3_state: Dict[str, torch.Tensor], *, strict: bool = True
                        ) -> Tuple[Dict[str, torch.Tensor], MappingReport]:
    mapped = dict(x3_state)
    missing, mismatches = [], []
    copied = 0
    targets = set()
    for base_key, base_tensor in base_state.items():
        target_key = x3_key_for_base_key(base_key)
        targets.add(target_key)
        if target_key not in x3_state:
            missing.append(f"{base_key} -> {target_key}")
            continue
        target_tensor = x3_state[target_key]
        if tuple(target_tensor.shape) != tuple(base_tensor.shape):
            mismatches.append(f"{base_key}: base={tuple(base_tensor.shape)} x3={tuple(target_tensor.shape)}")
            continue
        # Keep a view/reference: cloning a 3B checkpoint would double peak RAM.
        mapped[target_key] = base_tensor.detach()
        copied += 1
    unexpected = sorted(k for k in x3_state if k.startswith("backbone.")
                        and ".adapter." not in k and k not in targets)
    report = MappingReport(copied, missing, unexpected, mismatches)
    if strict and not report.ok:
        raise RuntimeError(f"Strict Qwen->X3 mapping failed: {report.to_dict()}")
    return mapped, report
