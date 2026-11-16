from __future__ import annotations

from pathlib import Path

IGNORED = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "models", "bin"}


def scan_project(root: str | Path, max_files: int = 500) -> list[str]:
    base = Path(root).resolve()
    files: list[str] = []
    for p in base.rglob("*"):
        if any(part in IGNORED for part in p.parts) or not p.is_file():
            continue
        try:
            resolved = p.resolve()
            if resolved == base or base not in resolved.parents:
                continue
            if resolved.stat().st_size <= 1_000_000:
                files.append(p.relative_to(base).as_posix())
        except OSError:
            continue
        if len(files) >= max_files:
            break
    return sorted(files)


def project_context(root: str | Path, max_chars: int = 12000) -> str:
    base = Path(root).resolve()
    output: list[str] = []
    used = 0
    for rel in scan_project(base):
        p = (base / rel).resolve()
        if base not in p.parents:
            continue
        block = f"\n### {rel}\n"
        try:
            if p.suffix.lower() not in {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".js", ".ts", ".ps1"}:
                continue
            block += p.read_text(encoding="utf-8", errors="replace") + "\n"
        except OSError:
            continue
        if used + len(block) > max_chars:
            break
        output.append(block)
        used += len(block)
    return "".join(output)
