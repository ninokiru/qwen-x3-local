# M2 test report

Environment: Python 3.13.5, PyTorch 2.10.0+cpu, pytest 9.0.2. Transformers is not installed. No Qwen checkpoint or GPU was used.

Command: `pytest -q`

Result: **16 passed, 1 skipped**.

The skipped test is the opt-in real-checkpoint integration test (`X3_RUN_REAL_CHECKPOINT=1`). The passing tests use synthetic tiny modules and cover zero-adapter logits parity, explicit decoder wrapping, tuple/cache passthrough, forward output/loss/metrics, backward gradients, wrapped-key mapping, save/load round-trip, generation delegation, adapter identity, router balance/collapse telemetry, strict wrapped-key mapping, save/load round-trip, and single-file/indexed safetensors header-audit fixtures.

Not tested: real 3B tensor mapping, actual tied embedding storage, actual tokenizer/vocab compatibility, exact Transformers cache signature, CUDA/BF16 behavior, training convergence, GGUF conversion or llama.cpp execution.
