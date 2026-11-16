from __future__ import annotations

from pathlib import Path

from .project_scanner import IGNORED


def search_code(root: str | Path, query: str, max_results: int = 100) -> list[str]:
    base = Path(root).resolve()
    results: list[str] = []
    if not query.strip():
        return results
    for p in base.rglob("*"):
        if not p.is_file() or any(part in IGNORED for part in p.parts):
            continue
        try:
            resolved = p.resolve()
            if resolved == base or base not in resolved.parents:
                continue
            if resolved.stat().st_size > 500_000:
                continue
            lines = resolved.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for no, line in enumerate(lines, 1):
            if query.casefold() in line.casefold():
                results.append(f"{p.relative_to(base).as_posix()}:{no}: {line.strip()[:240]}")
                if len(results) >= max_results:
                    return results
    return results
