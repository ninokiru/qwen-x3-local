from qwen_coder_x3.configuration_x3 import X3Config


def test_config_defaults():
    cfg = X3Config()
    assert cfg.num_experts == 4
    assert cfg.adapter_bottleneck_size == 32
    assert cfg.expert_names == ("coding", "reasoning", "agent", "multilingual")
