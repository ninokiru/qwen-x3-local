"""Cấu hình ứng dụng. Đọc từ configs/app.json; thiếu file thì dùng giá trị mặc định."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ServerConfig:
    base_url: str = "http://127.0.0.1:8080"
    timeout_s: int = 300
    temperature: float = 0.2
    max_tokens: int = 1024


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    workspace: str = "workspace"
    max_steps: int = 6
    max_history: int = 40
    run_timeout_s: int = 20
    db_path: str = "data/memory.sqlite3"


def load_config(path: str | Path | None) -> AppConfig:
    if path is None or not Path(path).exists():
        return AppConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    server = ServerConfig(**raw.get("server", {}))
    rest = {k: v for k, v in raw.items() if k != "server"}
    return AppConfig(server=server, **rest)
