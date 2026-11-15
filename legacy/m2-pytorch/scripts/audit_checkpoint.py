import argparse
import json
from qwen_coder_x3.checkpoint_audit import audit_checkpoint

parser = argparse.ArgumentParser(description="Low-memory Qwen2 safetensors checkpoint audit")
parser.add_argument("checkpoint_dir", help="Local directory containing config.json and safetensors index/shards")
args = parser.parse_args()
print(json.dumps(audit_checkpoint(args.checkpoint_dir), indent=2))
