import argparse, json
from transformers import AutoConfig

EXPECTED = {
    "hidden_size": 2048,
    "intermediate_size": 11008,
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    args = p.parse_args()
    cfg = AutoConfig.from_pretrained(args.model)
    data = cfg.to_dict()
    print(json.dumps({k: data.get(k) for k in EXPECTED}, indent=2))
    errors = []
    for key, expected in EXPECTED.items():
        actual = data.get(key)
        if actual != expected:
            errors.append(f"{key}: expected {expected}, got {actual}")
    if errors:
        raise SystemExit("Checkpoint config differs from M1 assumptions:\n" + "\n".join(errors))
    print("Checkpoint config matches X3 M1 baseline.")

if __name__ == "__main__":
    main()
