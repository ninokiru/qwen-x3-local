from .configuration_x3 import X3Config

__all__ = ["X3Config", "X3ForCausalLM"]


def __getattr__(name):
    if name == "X3ForCausalLM":
        from .modeling_x3 import X3ForCausalLM
        return X3ForCausalLM
    raise AttributeError(name)
