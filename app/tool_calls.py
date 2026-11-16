"""Phân tích và tạo khối gọi công cụ theo đúng định dạng dùng khi train:

    <tool_call>
    {"name": "read_file", "arguments": {"path": "config.yaml"}}
    </tool_call>
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

MAX_TOOL_CALLS = 4
MAX_TOOL_JSON_CHARS = 16000
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass
class ParsedReply:
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_reply(content: str) -> ParsedReply:
    calls: list[ToolCall] = []
    errors: list[str] = []
    for match in TOOL_CALL_RE.finditer(content):
        if len(calls) >= MAX_TOOL_CALLS:
            errors.append(f"Vượt quá {MAX_TOOL_CALLS} tool call trong một lượt")
            break
        raw = match.group(1)
        if len(raw) > MAX_TOOL_JSON_CHARS:
            errors.append("Tool call vượt giới hạn kích thước")
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"JSON không hợp lệ: {exc.msg}")
            continue
        if not isinstance(obj, dict):
            errors.append("Khối tool_call phải là một object JSON")
            continue
        name = obj.get("name")
        args = obj.get("arguments", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            errors.append("Thiếu 'name' (chuỗi) hoặc 'arguments' (object)")
            continue
        calls.append(ToolCall(name, args))
    text = TOOL_CALL_RE.sub("", content).strip()
    return ParsedReply(text=text, calls=calls, errors=errors)


def tool_prompt(tool_specs: list[dict]) -> str:
    """Tạo phần mô tả công cụ đưa vào system prompt."""
    lines = [
        "Bạn có thể gọi công cụ. Khi cần, trả về đúng một khối theo mẫu:",
        "<tool_call>",
        '{"name": "<tên công cụ>", "arguments": {...}}',
        "</tool_call>",
        "",
        "Danh sách công cụ:",
    ]
    for spec in tool_specs:
        params = json.dumps(spec["parameters"], ensure_ascii=False)
        approval = " | cần người dùng phê duyệt" if spec.get("requires_approval") else ""
        lines.append(f"- {spec['name']}: {spec['description']} | tham số: {params}{approval}")
    return "\n".join(lines)
