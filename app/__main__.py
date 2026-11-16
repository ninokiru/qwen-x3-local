"""CLI và Web UI của Qwen X3 Local."""
from __future__ import annotations

import argparse
from pathlib import Path

from .agent import Agent
from .config import load_config
from .llm_client import LLMError, LlamaServerClient
from .memory import Memory
from .rag import DocumentIndexer
from .router import classify
from .tools import Tool, Workspace, build_tools


def _approval(tool: Tool, arguments: dict) -> bool:
    print("\n[YÊU CẦU PHÊ DUYỆT]")
    print(f"Công cụ: {tool.name}")
    if tool.name == "run_python":
        print("CẢNH BÁO: Python có thể đọc/ghi file và truy cập mạng theo quyền của user Windows.")
    for key, value in arguments.items():
        if key == "code":
            value = str(value)[:600] + ("..." if len(str(value)) > 600 else "")
        elif key == "content":
            value = f"<nội dung {len(str(value))} ký tự>"
        print(f"  {key}: {value}")
    try:
        return input("Cho phép thực hiện đúng thao tác này? [y/N] ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        print("Không có xác nhận; đã từ chối thao tác.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen X3 Local")
    parser.add_argument("--config", default="configs/app.json")
    parser.add_argument("--web", action="store_true", help="Chạy Web UI thay vì CLI")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ws = Workspace(cfg.workspace, run_timeout_s=cfg.run_timeout_s,
                   allow_python=cfg.allow_unsafe_python)
    memory = Memory(cfg.db_path)
    tools = build_tools(ws)
    client = LlamaServerClient(cfg.server)
    agent = Agent(client, tools, max_steps=cfg.max_steps, max_history=cfg.max_history,
                  approval_callback=_approval)
    indexer = DocumentIndexer(memory)

    if args.web:
        if not cfg.web.enabled:
            raise SystemExit("Web UI đang tắt trong configs/app.json")
        from .web import serve
        try:
            serve(agent, memory, cfg)
        finally:
            memory.close()
        return

    conversation_id = memory.create_conversation("CLI session")
    print("Qwen X3 Local v1.0. Gõ /help để xem lệnh, /exit để thoát.")
    print(f"llama-server: {cfg.server.base_url} | context mục tiêu: {cfg.runtime.context_size}")
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
        if line == "/help":
            print("/add <nội dung>  lưu ghi nhớ | /memories  xem ghi nhớ | /forget <id>  xóa ghi nhớ")
            print("/search <từ khóa>  tìm tài liệu | /reset  xóa context chat | /web  mở Web UI")
            continue
        if line == "/reset":
            agent.reset()
            conversation_id = memory.create_conversation("CLI session mới")
            print("Đã tạo phiên hội thoại mới.")
            continue
        if line.startswith("/add "):
            memory.save_memory("note", line[5:])
            print("Đã lưu ghi nhớ.")
            continue
        if line == "/memories":
            for mid, kind, text in memory.list_memories():
                print(f"[{mid}] ({kind}) {text}")
            continue
        if line.startswith("/forget "):
            try:
                deleted = memory.delete_memory(int(line.split(maxsplit=1)[1]))
                print("Đã xóa." if deleted else "Không tìm thấy ghi nhớ.")
            except (ValueError, IndexError):
                print("Cú pháp: /forget <id>")
            continue
        if line.startswith("/search "):
            hits = memory.search(line[8:])
            for src, text in hits:
                print(f"[{src}] {text[:600]}\n")
            if not hits:
                print("Không tìm thấy.")
            continue
        if line == "/web":
            from .web import serve
            try:
                serve(agent, memory, cfg)
            except KeyboardInterrupt:
                pass
            continue

        memory.save_message(conversation_id, "user", line)
        context = indexer.retrieve(line, cfg.memory.top_k, cfg.memory.max_context_chars) if cfg.memory.enabled else ""
        try:
            answer = agent.ask(line, extra_context=context or None)
            print(f"\nAI> {answer}")
            memory.save_message(conversation_id, "assistant", answer)
        except LLMError as exc:
            print(f"\n[lỗi] {exc}")

    memory.close()


if __name__ == "__main__":
    main()
