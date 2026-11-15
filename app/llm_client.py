"""Client mỏng gọi llama-server (API tương thích OpenAI). Chỉ dùng thư viện chuẩn."""
from __future__ import annotations

import json
import urllib.request

from .config import ServerConfig


class LLMError(RuntimeError):
    pass


class LlamaServerClient:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg

    def chat(self, messages: list[dict]) -> str:
        body = json.dumps({
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.base_url.rstrip("/") + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except OSError as exc:
            raise LLMError(f"Không kết nối được llama-server: {exc}") from exc
        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Phản hồi không đúng định dạng: {payload!r}") from exc
