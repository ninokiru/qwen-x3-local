from __future__ import annotations

from typing import Any
from torch import nn

from .adapters import RoutedAdapterBank


class X3DecoderLayer(nn.Module):
    """Explicit composition: unchanged Qwen2 block, then one routed adapter."""

    def __init__(self, base_layer: nn.Module, adapter: RoutedAdapterBank):
        super().__init__()
        self.base_layer = base_layer
        self.adapter = adapter

    def forward(self, *args: Any, **kwargs: Any):
        # Pass attention_mask, position_ids, cache_position, position_embeddings,
        # past_key_value(s), output_attentions, etc. untouched to the exact
        # Transformers Qwen2 layer implementation. This avoids reimplementing
        # version-sensitive RoPE/GQA/cache semantics in the X3 wrapper.
        output = self.base_layer(*args, **kwargs)
        if isinstance(output, tuple):
            if not output:
                raise RuntimeError("Qwen2 decoder layer returned an empty tuple")
            return (self.adapter(output[0]),) + output[1:]
        if isinstance(output, list):
            if not output:
                raise RuntimeError("Qwen2 decoder layer returned an empty list")
            return [self.adapter(output[0]), *output[1:]]
        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = self.adapter(output.last_hidden_state)
            return output
        if hasattr(output, "shape"):
            return self.adapter(output)
        raise TypeError(f"Unsupported Qwen2 decoder output type: {type(output)!r}")
