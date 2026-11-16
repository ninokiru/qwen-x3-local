from __future__ import annotations

from pathlib import Path

from .chunker import split_text
from .loader import load_bytes, load_path


class DocumentIndexer:
    def __init__(self, memory, chunk_chars: int = 1800, overlap: int = 200):
        self.memory = memory
        self.chunk_chars = chunk_chars
        self.overlap = overlap

    def index_bytes(self, name: str, data: bytes, source: str | None = None) -> int:
        text = load_bytes(name, data)
        chunks = split_text(text, self.chunk_chars, self.overlap)
        identity = source or Path(name).name
        self.memory.remove_document(identity)
        for i, chunk in enumerate(chunks, 1):
            self.memory.add(f"document:{identity}#{i}", chunk)
        return len(chunks)

    def index_path(self, path: str | Path, source: str | None = None) -> int:
        p = Path(path)
        return self.index_bytes(p.name, p.read_bytes(), source=source or p.as_posix())

    def retrieve(self, query: str, k: int = 4, max_chars: int = 5000) -> str:
        hits = self.memory.search(query, k=k)
        selected: list[str] = []
        used = 0
        for source, text in hits:
            block = f"[Nguồn: {source}]\n{text}"
            if used + len(block) > max_chars:
                block = block[:max(0, max_chars - used)]
            if block:
                selected.append(block)
                used += len(block)
            if used >= max_chars:
                break
        return "\n\n---\n\n".join(selected)
