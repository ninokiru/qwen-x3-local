from types import SimpleNamespace
import torch
from torch import nn

from qwen_coder_x3.configuration_x3 import X3Config
from qwen_coder_x3.modeling_x3 import X3ForCausalLM
from qwen_coder_x3.decoder_layer import X3DecoderLayer
from qwen_coder_x3.weight_mapping import map_base_state_dict


class TinyQwenBlock(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        # Simulate tuple/cache return shape from a Qwen2 decoder layer.
        return (hidden_states + self.proj(hidden_states), "cache-sentinel")


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            vocab_size=32, max_position_embeddings=64, rope_theta=10000.0,
            rms_norm_eps=1e-6, tie_word_embeddings=True)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyQwenBlock(), TinyQwenBlock()])
        self.embed = nn.Embedding(32, 16)
        self.lm = nn.Linear(16, 32, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                use_cache=False, past_key_values=None, labels=None, **kwargs):
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            result = layer(hidden, attention_mask=attention_mask, position_ids=position_ids,
                           use_cache=use_cache, past_key_values=past_key_values)
            hidden = result[0]
        logits = self.lm(hidden)
        loss = logits.square().mean() if labels is not None else None
        return SimpleNamespace(logits=logits, loss=loss,
            past_key_values=("tiny-cache",) if use_cache else None,
            hidden_states=None, attentions=None)

    def generate(self, input_ids, **kwargs):
        return input_ids


def tiny_x3():
    return X3ForCausalLM(TinyBackbone(), X3Config(adapter_bottleneck_size=4))


def test_explicit_layer_wraps_base_and_preserves_tuple_tail():
    block = TinyQwenBlock()
    from qwen_coder_x3.adapters import RoutedAdapterBank
    wrapper = X3DecoderLayer(block, RoutedAdapterBank(16, 4,
        ("coding", "reasoning", "agent", "multilingual")))
    x = torch.randn(2, 3, 16)
    y = wrapper(x)
    assert y[0].shape == x.shape
    assert y[1] == "cache-sentinel"


def test_base_vs_x3_logits_parity_zero_adapters():
    torch.manual_seed(4)
    base = TinyBackbone()
    ids = torch.randint(0, 32, (2, 5))
    ref_logits = base(ids).logits.detach()
    x3 = X3ForCausalLM(base, X3Config(adapter_bottleneck_size=4))
    # All expert up-projections are zero initialized, so X3 must be identity
    # around each base decoder block at initialization.
    out = x3(ids)
    assert torch.allclose(out.logits, ref_logits, atol=1e-6, rtol=1e-6)


def test_forward_exposes_loss_aux_metrics_and_cache_passthrough():
    model = tiny_x3()
    ids = torch.randint(0, 32, (2, 4))
    out = model(ids, attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(4).unsqueeze(0), use_cache=True,
                past_key_values=None, labels=ids)
    assert out.logits.shape == (2, 4, 32)
    assert out.loss is not None and out.aux_router_loss.ndim == 0
    assert "router_entropy" in out.router_metrics
    assert out.past_key_values == ("tiny-cache",)
    assert out.total_loss.ndim == 0


def test_backward_reaches_experts_and_router():
    model = tiny_x3()
    ids = torch.randint(0, 32, (2, 4))
    out = model(ids, labels=ids)
    out.total_loss.backward()
    adapter = model.backbone.model.layers[0].adapter
    assert adapter.experts[0].up.weight.grad is not None
    assert adapter.router.weight.grad is not None


def test_mapping_uses_wrapped_layer_key_names():
    base = TinyBackbone()
    base_state = base.state_dict()
    model = X3ForCausalLM(base, X3Config(adapter_bottleneck_size=4))
    state, report = map_base_state_dict(base_state, model.state_dict(), strict=True)
    assert report.ok and report.copied == len(base_state)
    assert torch.equal(state["backbone.model.layers.0.base_layer.proj.weight"],
                       base_state["model.layers.0.proj.weight"])


def test_save_load_round_trip(tmp_path):
    model = tiny_x3()
    model.save_x3_pretrained(str(tmp_path))
    saved = torch.load(tmp_path / "x3_m2.pt", map_location="cpu", weights_only=False)
    assert saved["format"] == "qwen-coder-x3-m2"
    clone = tiny_x3()
    clone.load_state_dict(saved["state_dict"])
    for key, value in model.state_dict().items():
        assert torch.equal(value, clone.state_dict()[key])


def test_generate_delegates_to_backbone():
    model = tiny_x3()
    ids = torch.tensor([[1, 2, 3]])
    assert torch.equal(model.generate(ids), ids)
