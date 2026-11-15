import torch
from qwen_coder_x3.adapters import RoutedAdapterBank


def test_adapter_is_identity_at_initialization():
    torch.manual_seed(7)
    bank = RoutedAdapterBank(32, 8, ("coding", "reasoning", "agent", "multilingual"))
    x = torch.randn(2, 5, 32)
    assert torch.allclose(bank(x), x, atol=1e-7)


def test_adapter_gradients_flow():
    bank = RoutedAdapterBank(32, 8, ("coding", "reasoning", "agent", "multilingual"))
    x = torch.randn(2, 5, 32, requires_grad=True)
    (bank(x).square().mean() + bank.auxiliary_loss()).backward()
    assert x.grad is not None
    assert bank.router.weight.grad is not None
    assert bank.experts[0].up.weight.grad is not None


def test_router_uniform_initialization_has_balanced_utilization():
    bank = RoutedAdapterBank(32, 8, ("coding", "reasoning", "agent", "multilingual"))
    bank(torch.randn(2, 7, 32))
    utilization = bank.last_metrics["expert_utilization"]
    assert torch.allclose(utilization, torch.full((4,), 0.25), atol=1e-6)
    assert bank.last_metrics["expert_collapse"].item() < 1e-6
    assert bank.last_metrics["router_entropy"].item() > 1.3


def test_router_collapse_metric_detects_single_expert_bias():
    bank = RoutedAdapterBank(16, 4, ("coding", "reasoning", "agent", "multilingual"))
    with torch.no_grad():
        bank.router.bias[0] = 12.0
    bank(torch.randn(2, 3, 16))
    utilization = bank.last_metrics["expert_utilization"]
    assert utilization[0] > 0.99
    assert bank.last_metrics["expert_collapse"].item() > 0.7
