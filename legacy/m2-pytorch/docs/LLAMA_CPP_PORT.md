# llama.cpp / GGML native port design — M5/M6 (not implemented)

Stock Qwen2 GGUF cannot execute X3 adapter/router operations. Do not relabel an X3 file as `qwen2`; that would omit model computation.

## Proposed GGUF tensors

For each layer `blk.N`:
- `x3_blk.N.router_norm.weight`, `x3_blk.N.router_norm.bias`
- `x3_blk.N.router.weight`, `x3_blk.N.router.bias`
- `x3_blk.N.expert.E.norm.weight`, `x3_blk.N.expert.E.norm.bias`
- `x3_blk.N.expert.E.down.weight`
- `x3_blk.N.expert.E.up.weight`

Metadata should include a versioned architecture id (`qwen_coder_x3`), X3 schema revision, expert count and stable expert ordering, bottleneck size, router temperature, adapter scale, load-balance/entropy training coefficients (training metadata only), injection location, base Qwen2 architecture parameters, RoPE parameters, GQA head counts, vocab size and tied-embedding flag. Never serialize a metadata value that contradicts tensor shapes.

## Graph operations per decoder block

1. Execute the existing Qwen2 block unchanged (attention/GQA/RoPE/MLP/norm/cache).
2. Apply router RMS/LayerNorm then linear projection and temperature scaling.
3. Softmax over four experts per token.
4. For each expert: norm -> down projection -> GELU -> up projection.
5. Multiply each expert output by its token gate, sum experts, scale, residual-add to block hidden state.

Inference graph does not need the auxiliary load-balancing loss. Training graph may expose router probabilities and auxiliary statistics separately.

## Quantization

- First parity baseline: F16/F32, no quantization.
- Then test Q8_0 and Q5_K_M, followed by Q4_K_M.
- Quantize backbone and X3 matrices explicitly; router logits/norm and small expert matrices may need higher precision if routing drift is observed.
- Record per-layer max absolute/relative error, logits KL, top-token agreement, perplexity delta and task benchmark delta.

## Required tests

- GGUF round-trip tensor-name/shape coverage; no missing or extra X3 tensors.
- PyTorch-vs-GGML layer parity with adapters zeroed and with nonzero randomized adapters.
- Router probability/utilization parity and deterministic expert ordering.
- Full logits parity at F16 before quantization; cache and multi-token decode parity.
- Quantized quality regression thresholds on code, reasoning, tool-call and multilingual suites.

M5: converter, metadata, loader and F16 CPU graph. M6: optimized kernels, cache integration, quantization and end-to-end parity. No GGUF runtime support exists in M2.
