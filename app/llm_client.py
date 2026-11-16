"""Client llama-server tương thích OpenAI, chỉ dùng thư viện chuẩn Python."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .config import ServerConfig


class LLMError(RuntimeError):
    pass


class LlamaServerClient:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg

    def _request(self, path: str, data: dict | None = None, timeout_s: int | None = None) -> dict:
        url = self.cfg.base_url.rstrip("/") + path
        body = None if data is None else json.dumps(data, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"} if body is not None else {},
            method="POST" if body is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s or self.cfg.timeout_s) as resp:
                raw = resp.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise LLMError("Phản hồi từ llama-server vượt giới hạn 8 MB")
            return json.loads(raw.decode("utf-8")) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", errors="replace")
            raise LLMError(f"llama-server trả HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LLMError(f"Không gọi được llama-server: {exc}") from exc

    def health(self) -> bool:
        try:
            self._request("/health", timeout_s=min(self.cfg.timeout_s, 3))
            return True
        except LLMError:
            return False

    def chat(self, messages: list[dict]) -> str:
        body = {
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
        }
        attempts = max(0, min(self.cfg.retries, 2)) + 1
        last_error: LLMError | None = None
        for attempt in range(attempts):
            try:
                payload = self._request("/v1/chat/completions", body)
                return payload["choices"][0]["message"].get("content") or ""
            except LLMError as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(0.25 * (attempt + 1))
        assert last_error is not None
        raise last_error
