from app.memory import Memory
from app.rag.chunker import split_text
from app.rag.pipeline import DocumentIndexer


def test_split_text_keeps_all_content():
    text = "abc " * 1000
    chunks = split_text(text, chunk_chars=500, overlap=40)
    assert len(chunks) > 1
    assert all(chunks)


def test_index_and_retrieve_markdown():
    mem = Memory()
    indexer = DocumentIndexer(mem, chunk_chars=500, overlap=50)
    count = indexer.index_bytes("guide.md", b"# Qwen X3\n\nCau hinh local offline va SQLite FTS5.", source="guide.md")
    assert count == 1
    result = indexer.retrieve("SQLite FTS5", k=3)
    assert "guide.md" in result and "SQLite" in result
    mem.close()


def test_reindex_replaces_chunks():
    mem = Memory()
    idx = DocumentIndexer(mem)
    idx.index_bytes("a.txt", b"old sentence", source="a.txt")
    idx.index_bytes("a.txt", b"new sentence", source="a.txt")
    assert "new sentence" in idx.retrieve("new sentence")
    assert not idx.retrieve("old sentence")
    mem.close()
