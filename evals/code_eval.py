"""Đánh giá câu trả lời code trên một bộ test cố định.

Mỗi case trong evals/cases/code.jsonl:
  {"question": "...", "keywords": ["def", "return"], "min_code_blocks": 1}

Chỉ số:
  has_code     - có ít nhất min_code_blocks khối ``` trong câu trả lời
  has_keywords - có đủ mọi từ khóa trong câu trả lời, không phân biệt hoa thường
  no_tool      - KHÔNG lẫn khối <tool_call> (chế độ code không có công cụ)

Chạy (llama-server đang chạy ở cổng 8080):
    python -m evals.code_eval --cases evals/cases/code.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.agent import BASE_SYSTEM
from app.config import ServerConfig
from app.llm_client import LlamaServerClient
from app.tool_calls import parse_reply

FENCE_RE = re.compile(r"```")
METRICS = ("has_code", "has_keywords", "no_tool")


def score_case(content: str, keywords: list[str], min_code_blocks: int = 1) -> dict:
    blocks = len(FENCE_RE.findall(content)) // 2
    low = content.lower()
    return {
        "has_code": blocks >= min_code_blocks,
        "has_keywords": all(k.lower() in low for k in keywords),
        "no_tool": not parse_reply(content).calls,
    }


def run(cases_path: Path, base_url: str) -> dict:
    client = LlamaServerClient(ServerConfig(base_url=base_url, temperature=0.0))
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    totals = dict.fromkeys(METRICS, 0)
    for case in cases:
        messages = [{"role": "system", "content": BASE_SYSTEM}, {"role": "user", "content": case["question"]}]
        content = client.chat(messages)
        result = score_case(content, case.get("keywords", []), case.get("min_code_blocks", 1))
        for k in METRICS:
            totals[k] += result[k]
    n = len(cases) or 1
    return {k: v / n for k, v in totals.items()} | {"cases": len(cases)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evals/cases/code.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    report = run(Path(args.cases), args.base_url)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
