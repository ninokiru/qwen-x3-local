"""Định tuyến đơn giản theo từ khóa: chọn chế độ (chat / code / agent).

Dùng heuristic để giữ nhẹ. Khi có dữ liệu thật, có thể thay bằng bộ phân loại nhỏ.
"""
from __future__ import annotations

import re

AGENT_HINTS = ("đọc file", "ghi file", "liệt kê", "thư mục", "workspace", "chạy code", "chạy file")
CODE_HINTS = ("code", "hàm", "lỗi", "bug", "python", "class", "script", "debug", "test", "viết")


def _has(text: str, hints: tuple[str, ...]) -> bool:
    return any(re.search(r"(?<!\w)" + re.escape(h) + r"(?!\w)", text) for h in hints)


def route(message: str) -> str:
    m = message.lower()
    if _has(m, AGENT_HINTS):
        return "agent"
    if _has(m, CODE_HINTS):
        return "code"
    return "chat"


# Tên LoRA tương ứng với từng chế độ. None nghĩa là dùng model gốc.
MODE_TO_LORA: dict[str, str | None] = {"chat": None, "code": "code", "agent": "agent"}
