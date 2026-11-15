# Qwen-Coder-X3 M2 architecture

## Explicit composition

```text
input_ids / masks / positions / cache
                 |
        Original Qwen2 embedding
                 |
   +-------------v----------------+
   | Qwen2 decoder block 0        |
   | GQA + RoPE + RMSNorm + SwiGLU|
   +-------------+----------------+
                 |
       X3 routed adapter bank 0
                 |
             ... x 36
                 |
       Original final RMSNorm
                 |
       Original tied LM head
                 |
               logits
```

`X3DecoderLayer` owns `base_layer` and `adapter`. Its `forward(*args, **kwargs)` passes all arguments to the exact Transformers Qwen2 decoder layer, transforms only the first hidden-state output, and preserves remaining tuple/list values (including attention/cache outputs). Thus no second implementation of GQA/RoPE/cache semantics is introduced in PyTorch M2.

## Adapter

For token hidden state `h`:

`p = softmax(W_router LayerNorm(h) / temperature)`

`h' = h + scale * sum_e p_e Expert_e(h)`

Each expert is `Up(GELU(Down(LayerNorm(h))))`. `Up` is zero-initialized, so the adapter starts as an exact identity. Expert labels are objectives requiring data, not hard-coded capabilities.

## Router objective and telemetry

The load-balancing term is `E * sum_e mean(p_e)^2 - 1`, zero at uniform soft utilization. Optional entropy regularization is controlled independently. `X3Output` exposes mean router entropy, soft utilization per expert and utilization spread (`expert_collapse`). These metrics are diagnostics; soft utilization is not proof of semantic expert specialization.

## Checkpoint integrity

All backbone dimensions are read from the actual checkpoint config. The local header-only auditor checks Qwen2 model type, GQA divisibility, vocab/embedding/LM-head shapes, first-layer projection and MLP/norm shapes from safetensors headers. Strict mapping translates `model.layers.N.*` to `backbone.model.layers.N.base_layer.*` and other base keys to `backbone.*`. Adapter keys are explicitly excluded from pretrained tensor mapping. No base tensor is cloned during mapping, avoiding a second full 3B allocation.

M2 does not modify tokenizer files, vocabulary, embeddings or LM head. Tied embeddings must be confirmed on the actual loaded checkpoint before claiming real-checkpoint parity.
