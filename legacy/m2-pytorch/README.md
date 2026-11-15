# Qwen-Coder-X3 — M2 explicit decoder prototype

M2 replaces M1's forward hooks with explicit `X3DecoderLayer` modules. Each wrapper calls the original pretrained Qwen2 decoder block unchanged, then applies a token-wise routed residual adapter. The original block continues to own attention, GQA, RoPE, RMSNorm, SwiGLU, attention masks, position IDs and KV-cache behavior.

## Important architecture boundary

This is an explicit PyTorch composition around the upstream Qwen2 layer, **not yet a separately reimplemented Transformer kernel**. It preserves the exact base block implementation by delegation. Native GGML/llama.cpp support remains future work; a standard Qwen2 GGUF cannot execute X3 adapters.

The four expert names (`coding`, `reasoning`, `agent`, `multilingual`) identify training objectives only. They do not imply pre-existing skills. Each requires suitable supervised/trajectory data and evaluation.

## Hardware profiles

### Your PC: Pentium Gold G7400, 8 GB RAM, 240 GB SSD, Windows 11 26H2

Use this machine for source editing, Python environment setup, config inspection, CPU unit tests, small synthetic forward/backward tests and Git operations. The `test-lite` path uses tiny synthetic layers and does not download or load the 3B model.

Do **not** load/train the full 3B checkpoint on this PC. Even quantized inference is a separate deployment path and depends on runtime, context length and memory available; M2's PyTorch full-checkpoint conversion is not intended for this hardware. Keep at least several GB free on the SSD for Windows, Python environments and temporary files.

### Cloud GPU / high-memory host

Use this for real checkpoint load, full tensor audit, full-model parity, training, checkpoint export and later GGUF conversion. The full-load script requires an explicit `--allow-full-load` acknowledgement.

## Install and local lightweight tests (Windows PowerShell)

Python 3.10–3.12 is recommended for the local workflow. Create a virtual environment and install PyTorch CPU plus test dependencies. Install Transformers only when you need the real-checkpoint/cloud workflow.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch pytest
pip install -e . --no-deps
pytest -q
```

The project declares `torch>=2.3` and `transformers>=4.45`. In this implementation environment, unit tests were executed with **PyTorch 2.10.0+cpu** and Python 3.13.5; Transformers was not installed, so no Transformers version is claimed as tested. For a pinned full-checkpoint environment, choose and record the exact Transformers version before integration testing.

## Local checkpoint config/header audit (no model weights loaded)

Place a Hugging Face safetensors checkpoint locally, including `config.json`, `model.safetensors.index.json` and all referenced shards, then run:

```powershell
python scripts/audit_checkpoint.py D:\models\Qwen2.5-Coder-3B-Instruct
```

This reads config and tensor headers only. It does not load 3B parameters. The audit uses the checkpoint's actual dimensions; `configs/x3_lite.json` contains historical expected values for audit/reference only, never as overrides.

## Full checkpoint validation (cloud only)

```bash
python scripts/load_base_checkpoint.py \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --output ./artifacts/x3-m2 \
  --dtype bfloat16 \
  --allow-full-load
```

The script maps keys and shapes without cloning the entire base state dict. Run only on a host with sufficient RAM/GPU memory. The optional integration test uses `X3_CHECKPOINT_PATH` for a local checkpoint and is deliberately skipped by default.

## M2 API

`X3ForCausalLM.forward(...)` forwards all arguments unchanged to the Qwen2 causal LM, including `attention_mask`, `position_ids`, `use_cache`, `past_key_values` and `labels`. It returns `X3Output` with:
- `logits`
- base `loss` (when labels are supplied)
- `aux_router_loss`
- `total_loss` = base loss + auxiliary router loss
- `router_metrics`: mean router entropy, soft expert utilization and expert-collapse spread
- `past_key_values`, hidden states and attentions when returned by the selected Transformers implementation

The cache API is delegated rather than reimplemented. Compatibility must still be integration-tested against the exact Transformers version used for the real checkpoint.

## Tests

```powershell
pytest -q
```

`tests/test_real_checkpoint.py` is skipped unless `X3_RUN_REAL_CHECKPOINT=1` is set. Do not run that test on the 8 GB PC.

## Milestone status

- M2 explicit wrapper, router loss/metrics, key mapping and synthetic tests: implemented.
- Real Qwen checkpoint tensor audit, tied-embedding parity, Transformers cache compatibility: pending cloud/local checkpoint validation.
- Training data, expert specialization, DPO/distillation: not implemented.
- GGUF converter and native llama.cpp GGML graph: design only, not implemented.
