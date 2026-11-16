from __future__ import annotations

import shutil
from pathlib import Path


def check_local_install(root: str | Path = ".") -> dict[str, bool | str]:
    base = Path(root)
    server = base / "bin" / "llama-server.exe"
    model = base / "models" / "base-q4_k_m.gguf"
    return {
        "llama_server": server.is_file(),
        "base_model": model.is_file(),
        "python": shutil.which("python") is not None,
        "workspace": (base / "workspace").is_dir(),
    }
