"""Kiểm tra file model/LoRA và sinh tham số llama-server; không tự tải model."""
from __future__ import annotations

from pathlib import Path


class ModelManager:
    def __init__(self, root: str | Path = "."):
        self.root = Path(root).resolve()

    def resolve_model(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError("Model path nằm ngoài project")
        if not p.is_file() or p.suffix.lower() != ".gguf":
            raise FileNotFoundError(f"Không tìm thấy file GGUF: {p}")
        return p

    def server_args(self, model: str, context_size: int = 4096, threads: int = 4,
                    port: int = 8080, loras: list[str] | None = None) -> list[str]:
        model_path = self.resolve_model(model)
        args = ["-m", str(model_path), "-c", str(max(1024, min(context_size, 8192))),
                "-t", str(max(1, min(threads, 8))), "--host", "127.0.0.1",
                "--port", str(port)]
        for item in loras or []:
            args.extend(["--lora", str(self.resolve_model(item))])
        return args
