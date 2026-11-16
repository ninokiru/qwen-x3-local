"""Tạo context hội thoại gọn, không gọi model bổ sung mặc định."""
from __future__ import annotations


def compact_history(messages: list[dict[str, str]], max_messages: int = 12,
                    max_chars: int = 14000) -> list[dict[str, str]]:
    selected = messages[-max(2, max_messages):]
    out: list[dict[str, str]] = []
    used = 0
    for item in reversed(selected):
        content = str(item.get("content", ""))
        role = str(item.get("role", "user"))
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[:remaining]
        out.append({"role": role, "content": content})
        used += len(content)
    out.reverse()
    return out
