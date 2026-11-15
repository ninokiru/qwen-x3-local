import json
import pytest

safetensors_torch = pytest.importorskip("safetensors.torch")
from qwen_coder_x3.checkpoint_audit import audit_checkpoint


def test_header_only_checkpoint_audit_reads_actual_config_and_shapes(tmp_path):
    tensors = {
        "model.embed_tokens.weight": __import__("torch").zeros(32, 16),
        "model.layers.0.self_attn.q_proj.weight": __import__("torch").zeros(16, 16),
        "model.layers.0.self_attn.k_proj.weight": __import__("torch").zeros(8, 16),
        "model.layers.0.self_attn.v_proj.weight": __import__("torch").zeros(8, 16),
        "model.layers.0.mlp.gate_proj.weight": __import__("torch").zeros(32, 16),
        "model.layers.0.input_layernorm.weight": __import__("torch").ones(16),
    }
    safetensors_torch.save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen2", "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 1, "num_attention_heads": 4,
        "num_key_value_heads": 2, "vocab_size": 32,
        "max_position_embeddings": 64, "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6, "tie_word_embeddings": True,
    }), encoding="utf-8")
    weight_map = {key: "model.safetensors" for key in tensors}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8")

    report = audit_checkpoint(str(tmp_path))
    assert report["config"]["hidden_size"] == 16
    assert report["gqa"] == {"query_heads": 4, "kv_heads": 2}
    assert report["shape_checks"]["lm_head"]["tied_to_embeddings"] is True
    assert report["note"].startswith("Header-only")


def test_header_only_audit_accepts_single_safetensors_without_index(tmp_path):
    import torch
    tensors = {
        "model.embed_tokens.weight": torch.zeros(32, 16),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(16, 16),
        "model.layers.0.self_attn.k_proj.weight": torch.zeros(8, 16),
        "model.layers.0.self_attn.v_proj.weight": torch.zeros(8, 16),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros(32, 16),
        "model.layers.0.input_layernorm.weight": torch.ones(16),
    }
    safetensors_torch.save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen2", "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 1, "num_attention_heads": 4,
        "num_key_value_heads": 2, "vocab_size": 32,
        "max_position_embeddings": 64, "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6, "tie_word_embeddings": True,
    }), encoding="utf-8")
    report = audit_checkpoint(str(tmp_path))
    assert report["tensor_count"] == len(tensors)
