from __future__ import annotations

import difflib
from pathlib import Path


def make_unified_diff(path: str, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    ))


def preview_file_change(root: str | Path, relative_path: str, new_content: str) -> str:
    base = Path(root).resolve()
    target = (base / relative_path).resolve()
    if base not in target.parents:
        raise ValueError("Đường dẫn nằm ngoài project")
    old = target.read_text(encoding="utf-8") if target.exists() else ""
    return make_unified_diff(relative_path, old, new_content)
