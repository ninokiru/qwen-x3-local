"""Đánh giá khả năng gọi công cụ trên một bộ test cố định.

Mỗi case trong evals/cases/tool_call.jsonl:
  {"messages": [...], "expect_tool": "read_file" | null, "expect_args": {...}}
  expect_tool = null nghĩa là model KHÔNG được gọi công cụ.

Chạy (llama-server đang chạy ở cổng 8080):
    python -m evals.tool_call_eval --cases evals/cases/tool_call.jsonl
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from app.agent import BASE_SYSTEM
from app.config import ServerConfig
from app.llm_client import LlamaServerClient
from app.tool_calls import parse_reply, tool_prompt
from app.tools import Workspace, build_tools


def score_case(content: str, expect_tool: str | None, expect_args: dict | None) -> dict:
    parsed = parse_reply(content)
    if expect_tool is None:
        ok = not parsed.calls and not parsed.errors
        return {"format_ok": ok, "tool_ok": ok, "args_ok": ok}
    format_ok = bool(parsed.calls) and not parsed.errors
    if not format_ok:
        return {"format_ok": False, "tool_ok": False, "args_ok": False}
    first = parsed.calls[0]
    tool_ok = first.name == expect_tool
    args_ok = tool_ok and all(first.arguments.get(k) == v for k, v in (expect_args or {}).items())
    return {"format_ok": True, "tool_ok": tool_ok, "args_ok": args_ok}


def run(cases_path: Path, base_url: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tools = build_tools(Workspace(tmp))
        system = BASE_SYSTEM + "\n\n" + tool_prompt([t.spec() for t in tools.values()])
        client = LlamaServerClient(ServerConfig(base_url=base_url, temperature=0.0))
        cases = [json.loads(l) for l in cases_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        totals = {"format_ok": 0, "tool_ok": 0, "args_ok": 0}
        for case in cases:
            messages = [{"role": "system", "content": system}] + case["messages"]
            content = client.chat(messages)
            result = score_case(content, case.get("expect_tool"), case.get("expect_args"))
            for k in totals:
                totals[k] += result[k]
    n = len(cases) or 1
    return {k: v / n for k, v in totals.items()} | {"cases": len(cases)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evals/cases/tool_call.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    report = run(Path(args.cases), args.base_url)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
