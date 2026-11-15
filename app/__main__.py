"""CLI: python -m app --config configs/app.json"""
from __future__ import annotations

import argparse

from .agent import Agent
from .config import load_config
from .llm_client import LLMError, LlamaServerClient
from .memory import Memory
from .tools import Workspace, build_tools


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen X3 local agent")
    parser.add_argument("--config", default="configs/app.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ws = Workspace(cfg.workspace, run_timeout_s=cfg.run_timeout_s)
    memory = Memory(cfg.db_path)
    agent = Agent(
        LlamaServerClient(cfg.server),
        build_tools(ws),
        max_steps=cfg.max_steps,
        max_history=cfg.max_history,
    )

    print("Qwen X3 local. Gõ /exit để thoát, /reset để xóa hội thoại, /add <nội dung> để lưu ghi chú.")
    while True:
        try:
            line = input("\nBạn> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/exit":
            break
        if line == "/reset":
            agent.reset()
            print("Đã xóa hội thoại.")
            continue
        if line.startswith("/add "):
            memory.add("note", line[5:])
            print("Đã lưu ghi chú.")
            continue

        hits = memory.search(line)
        prompt = line
        if hits:
            context = "\n".join(f"[{src}] {txt}" for src, txt in hits)
            prompt = f"Tài liệu liên quan:\n{context}\n\nCâu hỏi: {line}"
        try:
            print(f"\nAI> {agent.ask(prompt)}")
        except LLMError as exc:
            print(f"\n[lỗi] {exc}")

    memory.close()


if __name__ == "__main__":
    main()
