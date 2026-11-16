"""SQLite memory: FTS5, ghi chú có loại, lịch sử hội thoại và migration nhẹ."""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path
from datetime import datetime, timezone


class Memory:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(
            source, text, tokenize='unicode61 remove_diacritics 1')""")
        self.db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS docs_norm USING fts5(
            source, text, normalized, tokenize='unicode61')""")
        self.db.execute("CREATE TABLE IF NOT EXISTS memory_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        migrated = self.db.execute("SELECT value FROM memory_meta WHERE key='docs_norm_v1'").fetchone()
        if not migrated:
            rows = self.db.execute("SELECT source, text FROM docs").fetchall()
            for source, text in rows:
                self.db.execute("INSERT INTO docs_norm(source, text, normalized) VALUES (?, ?, ?)",
                                 (source, text, self._normalize(text)))
            self.db.execute("INSERT INTO memory_meta(key, value) VALUES ('docs_norm_v1', '1')")
        self.db.execute("""CREATE TABLE IF NOT EXISTS memories(
            id INTEGER PRIMARY KEY, kind TEXT NOT NULL, text TEXT NOT NULL,
            created_at TEXT NOT NULL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS conversations(
            id INTEGER PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE)""")
        self.db.commit()

    @staticmethod
    def _normalize(text: str) -> str:
        decomposed = unicodedata.normalize("NFD", text.casefold())
        plain = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
        return plain.replace("đ", "d")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def add(self, source: str, text: str) -> None:
        clean = str(text).strip()
        if not clean:
            return
        self.db.execute("INSERT INTO docs(source, text) VALUES (?, ?)", (str(source), clean))
        self.db.execute("INSERT INTO docs_norm(source, text, normalized) VALUES (?, ?, ?)",
                        (str(source), clean, self._normalize(clean)))
        self.db.commit()

    def save_memory(self, kind: str, text: str) -> int:
        clean = str(text).strip()
        if not clean:
            raise ValueError("Nội dung ghi nhớ không được rỗng")
        cur = self.db.execute(
            "INSERT INTO memories(kind, text, created_at) VALUES (?, ?, ?)",
            (kind[:40], clean, self._now()),
        )
        self.db.commit()
        self.add(f"memory:{kind}", clean)
        return int(cur.lastrowid)

    def list_memories(self, limit: int = 100) -> list[tuple[int, str, str]]:
        rows = self.db.execute(
            "SELECT id, kind, text FROM memories ORDER BY id DESC LIMIT ?", (max(1, min(limit, 500)),)
        ).fetchall()
        return [(int(a), str(b), str(c)) for a, b, c in rows]

    def delete_memory(self, memory_id: int) -> bool:
        row = self.db.execute("SELECT kind, text FROM memories WHERE id=?", (memory_id,)).fetchone()
        if not row:
            return False
        self.db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        # FTS có thể chứa bản sao cùng nguồn; chỉ xóa nếu không còn memory trùng.
        still_exists = self.db.execute(
            "SELECT 1 FROM memories WHERE kind=? AND text=? LIMIT 1", row
        ).fetchone()
        if not still_exists:
            self.db.execute("DELETE FROM docs WHERE source=? AND text=?", (f"memory:{row[0]}", row[1]))
            self.db.execute("DELETE FROM docs_norm WHERE source=? AND text=?", (f"memory:{row[0]}", row[1]))
        self.db.commit()
        return True

    def remove_document(self, source: str) -> None:
        pattern = f"document:{source}#%"
        self.db.execute("DELETE FROM docs WHERE source LIKE ?", (pattern,))
        self.db.execute("DELETE FROM docs_norm WHERE source LIKE ?", (pattern,))
        self.db.commit()

    def search(self, query: str, k: int = 4) -> list[tuple[str, str]]:
        terms = re.findall(r"[\w]+", self._normalize(query), flags=re.UNICODE)
        if not terms:
            return []
        # Quote từng token để ký tự FTS đặc biệt không phá truy vấn.
        match = " AND ".join('"' + t.replace('"', '""') + '"' for t in terms[:20])
        try:
            rows = self.db.execute(
                "SELECT source, text FROM docs_norm WHERE docs_norm MATCH ? ORDER BY rank LIMIT ?",
                (match, max(1, min(int(k), 20))),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(str(r[0]), str(r[1])) for r in rows]

    def create_conversation(self, title: str = "Cuộc trò chuyện mới") -> int:
        now = self._now()
        cur = self.db.execute(
            "INSERT INTO conversations(title, created_at, updated_at) VALUES (?, ?, ?)",
            (title[:120], now, now),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def save_message(self, conversation_id: int, role: str, content: str) -> None:
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("role không hợp lệ")
        now = self._now()
        self.db.execute(
            "INSERT INTO messages(conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        self.db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
        self.db.commit()

    def list_conversations(self, limit: int = 50) -> list[tuple[int, str, str]]:
        rows = self.db.execute(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
            (max(1, min(limit, 200)),),
        ).fetchall()
        return [(int(a), str(b), str(c)) for a, b, c in rows]

    def get_messages(self, conversation_id: int, limit: int = 100) -> list[dict[str, str]]:
        rows = self.db.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
            (conversation_id, max(1, min(limit, 500))),
        ).fetchall()
        return [{"role": str(r), "content": str(c)} for r, c in reversed(rows)]

    def close(self) -> None:
        self.db.close()
