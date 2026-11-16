from __future__ import annotations


def split_text(text: str, chunk_chars: int = 1800, overlap: int = 200) -> list[str]:
    text = "\n".join(line.rstrip() for line in text.replace("\x00", "").splitlines()).strip()
    if not text:
        return []
    chunk_chars = max(300, min(chunk_chars, 6000))
    overlap = max(0, min(overlap, chunk_chars // 3))
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            boundary = text.rfind("\n", start + chunk_chars // 2, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks
