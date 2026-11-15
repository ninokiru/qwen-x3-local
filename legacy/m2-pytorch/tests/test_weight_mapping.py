import torch
from qwen_coder_x3.weight_mapping import map_base_state_dict


def test_mapping_copies_backbone_and_keeps_x3_keys():
    base = {"model.embed.weight": torch.ones(2, 3), "lm_head.weight": torch.zeros(4, 3)}
    x3 = {
        "backbone.model.embed.weight": torch.empty(2, 3),
        "backbone.lm_head.weight": torch.empty(4, 3),
        "adapter_banks.0.router.weight": torch.randn(4, 3),
    }
    mapped, report = map_base_state_dict(base, x3, strict=True)
    assert report.ok
    assert report.copied == 2
    assert torch.equal(mapped["backbone.model.embed.weight"], base["model.embed.weight"])
    assert "adapter_banks.0.router.weight" in mapped


def test_mapping_rejects_shape_mismatch():
    base = {"x": torch.ones(2, 3)}
    x3 = {"backbone.x": torch.empty(3, 2)}
    try:
        map_base_state_dict(base, x3, strict=True)
    except RuntimeError:
        return
    raise AssertionError("Expected strict mapping to fail")
