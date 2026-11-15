"""Đánh giá câu trả lời chat trên một bộ test cố định.

Mỗi case trong evals/cases/chat.jsonl:
  {"question": "...", "topics": ["Hà Nội"]}

Chỉ số:
  has_topics     - có đủ mọi chủ đề cần nêu trong câu trả lời, không phân biệt hoa thường
  no_tool        - KHÔNG lẫn khối <tool_call> (chế độ chat không có công cụ)
  in_vietnamese  - câu trả lời có dấu tiếng Việt, dùng VI_CHARS của data.prepare_dataset

Chạy (llama-server đang chạy ở cổng 8080):
    python -m evals.chat_eval --cases evals/cases/chat.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agent import BASE_SYSTEM
from app.config import ServerConfig
from app.llm_client import LlamaServerClient
from app.tool_calls import parse_reply
from data.prepare_dataset import VI_CHARS, VI_MIN_HITS

METRICS = ("has_topics", "no_tool", "in_vietnamese")


def score_case(content: str, topics: list[str]) -> dict:
    low = content.lower()
    vi_hits = sum(1 for ch in content.lower() if ch in VI_CHARS)
    return {
        "has_topics": all(t.lower() in low for t in topics),
        "no_tool": not parse_reply(content).calls,
        "in_vietnamese": vi_hits >= VI_MIN_HITS,
    }


def run(cases_path: Path, base_url: str) -> dict:
    client = LlamaServerClient(ServerConfig(base_url=base_url, temperature=0.0))
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    totals = dict.fromkeys(METRICS, 0)
    for case in cases:
        messages = [{"role": "system", "content": BASE_SYSTEM}, {"role": "user", "content": case["question"]}]
        content = client.chat(messages)
        result = score_case(content, case.get("topics", []))
        for k in METRICS:
            totals[k] += result[k]
    n = len(cases) or 1
    return {k: v / n for k, v in totals.items()} | {"cases": len(cases)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evals/cases/chat.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    report = run(Path(args.cases), args.base_url)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
