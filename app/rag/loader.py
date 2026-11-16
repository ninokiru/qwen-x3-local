from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree

TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".json",
    ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".ps1", ".bat",
    ".sql", ".html", ".css", ".xml", ".csv", ".java", ".c", ".cpp", ".h",
}


def load_bytes(name: str, data: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return data.decode("utf-8-sig", errors="replace")
    if ext == ".docx":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for para in root.findall(".//w:p", ns):
            text = "".join(node.text or "" for node in para.findall(".//w:t", ns))
            if text.strip():
                paragraphs.append(text)
        return "\n".join(paragraphs)
    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # optional dependency
        except ImportError as exc:
            raise ValueError("Đọc PDF cần cài gói tùy chọn: pip install '.[documents]'") from exc
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Định dạng chưa hỗ trợ: {ext or '(không có phần mở rộng)'}")


def load_path(path: str | Path) -> str:
    p = Path(path)
    return load_bytes(p.name, p.read_bytes())
