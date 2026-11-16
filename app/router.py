"""Router heuristic nhẹ, ưu tiên tiếng Việt và không cần model phân loại riêng."""
from __future__ import annotations

import re
from dataclasses import dataclass

AGENT_HINTS = (
    "đọc file", "ghi file", "sửa file", "tạo file", "xóa file", "liệt kê",
    "thư mục", "workspace", "chạy code", "chạy file", "tìm file", "tìm trong",
    "đọc pdf", "đọc tài liệu", "đọc source", "mở file", "upload", "project",
    "sửa code", "sửa project", "tạo project", "thêm tính năng vào", "chỉnh sửa project",
)
CODE_HINTS = (
    "code", "hàm", "lỗi", "bug", "python", "javascript", "class", "script",
    "debug", "test", "viết chương trình", "sửa code", "compile", "exception",
)


def _has(text: str, hints: tuple[str, ...]) -> bool:
    return any(re.search(r"(?<!\w)" + re.escape(h) + r"(?!\w)", text) for h in hints)


@dataclass(frozen=True)
class Intent:
    mode: str
    confidence: float
    reason: str


def classify(message: str) -> Intent:
    m = message.lower().strip()
    if _has(m, AGENT_HINTS):
        return Intent("agent", 0.82, "yêu cầu thao tác file/công cụ")
    if _has(m, CODE_HINTS):
        return Intent("code", 0.78, "yêu cầu lập trình")
    return Intent("chat", 0.60, "mặc định hội thoại")


def route(message: str) -> str:
    return classify(message).mode


# Adapter mặc định được nạp bởi script chạy server. Đây không phải cơ chế hot-swap.
MODE_TO_LORA: dict[str, str | None] = {"chat": None, "code": "code", "agent": "agent"}
