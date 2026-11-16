"""Cấu hình tập trung, tương thích với cấu hình v0.3."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ServerConfig:
    base_url: str = "http://127.0.0.1:8080"
    timeout_s: int = 180
    temperature: float = 0.2
    max_tokens: int = 768
    retries: int = 1


@dataclass
class RuntimeConfig:
    context_size: int = 4096
    threads: int = 4
    offline_mode: bool = True
    max_request_chars: int = 20000


@dataclass
class WebConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    max_upload_mb: int = 10


@dataclass
class MemoryConfig:
    enabled: bool = True
    semantic_search: bool = False  # FTS5 mặc định; embedding không tự tải xuống.
    top_k: int = 4
    max_context_chars: int = 5000


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    web: WebConfig = field(default_factory=WebConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    workspace: str = "workspace"
    max_steps: int = 5
    max_history: int = 12
    run_timeout_s: int = 10
    db_path: str = "data/qwen_x3.sqlite3"
    allow_unsafe_python: bool = False


def _make(cls: type, raw: dict[str, Any] | None):
    raw = raw or {}
    allowed = cls.__dataclass_fields__.keys()
    return cls(**{k: v for k, v in raw.items() if k in allowed})


def load_config(path: str | Path | None) -> AppConfig:
    """Đọc JSON; bỏ qua khóa cũ/không biết để cấu hình cũ tiếp tục chạy."""
    if path is None or not Path(path).exists():
        return AppConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Cấu hình phải là một JSON object")
    known_nested = {"server", "runtime", "web", "memory"}
    rest = {k: v for k, v in raw.items() if k not in known_nested and k in AppConfig.__dataclass_fields__}
    return AppConfig(
        server=_make(ServerConfig, raw.get("server")),
        runtime=_make(RuntimeConfig, raw.get("runtime")),
        web=_make(WebConfig, raw.get("web")),
        memory=_make(MemoryConfig, raw.get("memory")),
        **rest,
    )
