"""Full checkpoint conversion/validation; run on a cloud GPU host, not an 8 GB PC."""
import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM
from qwen_coder_x3 import X3Config, X3ForCausalLM
from qwen_coder_x3.weight_mapping import map_base_state_dict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    p.add_argument("--output", default="./artifacts/x3-m2")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--device-map", default=None)
    p.add_argument("--allow-full-load", action="store_true",
                   help="Acknowledge this loads the entire 3B model; use cloud GPU/large RAM")
    args = p.parse_args()
    if not args.allow_full_load:
        p.error("Refusing to load the full 3B checkpoint by default. Run on a suitable cloud host and pass --allow-full-load.")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"Loading full checkpoint {args.model}; this is NOT the 8 GB local test path.")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    # Keep tensor references before wrapping; no full-size clone is made.
    original_state = base.state_dict()
    x3 = X3ForCausalLM(base, X3Config(base_model_name_or_path=args.model))
    _, report = map_base_state_dict(original_state, x3.state_dict(), strict=True)
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "mapping_report.json"), "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    x3.save_x3_pretrained(args.output)
    print(json.dumps({"mapping": report.to_dict(), "output": args.output,
                      "total_parameters": sum(p.numel() for p in x3.parameters())}, indent=2))


if __name__ == "__main__":
    main()
