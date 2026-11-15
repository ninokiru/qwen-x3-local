"""Low-memory audit of a local HF checkpoint without loading model weights."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

REQUIRED_CONFIG_FIELDS = (
    "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "vocab_size",
    "max_position_embeddings", "rope_theta", "rms_norm_eps", "tie_word_embeddings",
)


def audit_checkpoint(checkpoint_dir: str) -> Dict[str, Any]:
    root = Path(checkpoint_dir)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_CONFIG_FIELDS if key not in config]
    if missing:
        raise ValueError(f"config.json missing required fields: {missing}")
    if config["model_type"] != "qwen2":
        raise ValueError(f"Expected Qwen2 checkpoint, got model_type={config['model_type']!r}")
    if int(config["num_attention_heads"]) % int(config["num_key_value_heads"]):
        raise ValueError("Invalid GQA config: query heads must divide evenly by KV heads")

    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        if not weight_map:
            raise ValueError("Safetensors index has an empty weight_map")
        shard_keys: Dict[str, list] = {}
        for key, shard in weight_map.items():
            shard_keys.setdefault(shard, []).append(key)
    else:
        # Single-file HF exports often omit the shard index.
        single = root / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError("Expected model.safetensors or model.safetensors.index.json")
        shard_keys = {single.name: None}

    # Read only tensor headers/slice shapes; do not materialize 3B parameters.
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("Install safetensors to audit tensor shapes without loading weights") from exc
    shapes = {}
    for shard, keys in shard_keys.items():
        shard_path = root / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing safetensors shard: {shard_path}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            available = list(handle.keys()) if keys is None else keys
            actual_keys = set(handle.keys())
            for key in available:
                if key not in actual_keys:
                    raise ValueError(f"Index references missing tensor {key} in {shard}")
                shapes[key] = list(handle.get_slice(key).get_shape())

    hidden = int(config["hidden_size"])
    vocab = int(config["vocab_size"])
    checks = {
        "embed_tokens": ("model.embed_tokens.weight", [vocab, hidden]),
        "lm_head": ("lm_head.weight", [vocab, hidden]),
        "q_proj": ("model.layers.0.self_attn.q_proj.weight", [hidden, hidden]),
        "k_proj": ("model.layers.0.self_attn.k_proj.weight", None),
        "v_proj": ("model.layers.0.self_attn.v_proj.weight", None),
        "gate_proj": ("model.layers.0.mlp.gate_proj.weight",
                      [int(config["intermediate_size"]), hidden]),
        "input_layernorm": ("model.layers.0.input_layernorm.weight", [hidden]),
    }
    shape_report = {}
    for label, (key, expected) in checks.items():
        if key not in shapes:
            # Some checkpoints tie/omit lm_head in the safetensors index.
            if label == "lm_head" and config.get("tie_word_embeddings"):
                shape_report[label] = {"present": False, "tied_to_embeddings": True}
                continue
            raise ValueError(f"Required tensor missing from checkpoint: {key}")
        actual = shapes[key]
        if label in ("k_proj", "v_proj"):
            expected = [int(config["num_key_value_heads"]) * hidden // int(config["num_attention_heads"]), hidden]
        if expected is not None and actual != expected:
            raise ValueError(f"Shape mismatch for {key}: expected {expected}, got {actual}")
        shape_report[label] = {"present": True, "shape": actual}

    return {
        "checkpoint": str(root),
        "config": {key: config[key] for key in REQUIRED_CONFIG_FIELDS},
        "tensor_count": len(shapes),
        "shape_checks": shape_report,
        "tied_embeddings": bool(config.get("tie_word_embeddings", False)),
        "gqa": {"query_heads": int(config["num_attention_heads"]),
                "kv_heads": int(config["num_key_value_heads"])},
        "note": "Header-only audit; model weights were not loaded.",
    }
