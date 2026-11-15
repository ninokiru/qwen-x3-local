"""Vòng lặp agent: gọi model, đọc tool_call, chạy công cụ, đưa kết quả lại cho model."""
from __future__ import annotations

from .llm_client import LlamaServerClient
from .router import route
from .tool_calls import parse_reply, tool_prompt
from .tools import Tool, execute

BASE_SYSTEM = "Bạn là trợ lý AI nói tiếng Việt, giỏi lập trình. Trả lời ngắn gọn, chính xác."


class Agent:
    def __init__(self, client, tools: dict[str, Tool], max_steps: int = 6, max_history: int = 40):
        self.client = client
        self.tools = tools
        self.max_steps = max_steps
        self.max_history = max_history
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history.clear()

    def _system_prompt(self, mode: str) -> str:
        if mode == "agent":
            specs = [t.spec() for t in self.tools.values()]
            return BASE_SYSTEM + "\n\n" + tool_prompt(specs)
        return BASE_SYSTEM

    def ask(self, user_text: str) -> str:
        mode = route(user_text)
        self.history.append({"role": "user", "content": user_text})
        system = {"role": "system", "content": self._system_prompt(mode)}

        for _ in range(self.max_steps):
            messages = [system] + self.history[-self.max_history:]
            content = self.client.chat(messages)
            parsed = parse_reply(content)
            self.history.append({"role": "assistant", "content": content})

            if parsed.errors:
                self.history.append({
                    "role": "user",
                    "content": "Lỗi định dạng tool_call: " + "; ".join(parsed.errors)
                               + ". Hãy thử lại đúng định dạng.",
                })
                continue
            if not parsed.calls:
                return parsed.text

            for call in parsed.calls:
                ok, result = execute(self.tools, call.name, call.arguments)
                self.history.append({"role": "tool", "content": result if ok else f"LỖI: {result}"})

        return "Đã đạt giới hạn số bước gọi công cụ."
