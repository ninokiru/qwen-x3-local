"""Bộ nhớ tài liệu đơn giản bằng SQLite FTS5.

unicode61 có remove_diacritics, nên truy vấn không dấu vẫn khớp văn bản có dấu.
Bước sau có thể thay bằng embedding (multilingual-e5-small) để tìm theo ngữ nghĩa.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path


class Memory:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(source, text, tokenize='unicode61 remove_diacritics 1')"
        )

    def add(self, source: str, text: str) -> None:
        self.db.execute("INSERT INTO docs(source, text) VALUES (?, ?)", (source, text))
        self.db.commit()

    def search(self, query: str, k: int = 3) -> list[tuple[str, str]]:
        terms = re.findall(r"\w+", query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        rows = self.db.execute(
            "SELECT source, text FROM docs WHERE docs MATCH ? ORDER BY rank LIMIT ?",
            (match, k),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def close(self) -> None:
        self.db.close()
