# M2 pretrained tensor mapping report

| Hugging Face base key | X3 M2 key |
|---|---|
| `model.embed_tokens.*` | `backbone.model.embed_tokens.*` |
| `model.layers.N.*` | `backbone.model.layers.N.base_layer.*` |
| `model.norm.*` | `backbone.model.norm.*` |
| `lm_head.*` | `backbone.lm_head.*` |

The mapping is strict: every source key must exist at the mapped destination and have identical shape. Shape mismatch, missing destination or unexpected non-adapter backbone tensor causes failure. X3-only tensors live under `backbone.model.layers.N.adapter.*` and are not treated as pretrained weights.

Mapping stores detached tensor references rather than cloning them. This is intentional: cloning a multi-billion-parameter state dict would create a large avoidable peak-RAM allocation. In the real conversion script, the base module is loaded once, then wrapped in-place while retaining the same Parameter objects.

**Not yet verified on the actual Qwen checkpoint in this environment.** The synthetic mapping test passes; a real report requires a local/cloud copy of the checkpoint and the exact Transformers version.
