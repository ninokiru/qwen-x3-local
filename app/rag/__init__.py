"""RAG local dựa trên SQLite FTS5; không tự tải embedding model."""
from .pipeline import DocumentIndexer

__all__ = ["DocumentIndexer"]
