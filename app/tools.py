"""Công cụ cho agent. Mọi thao tác file bị giới hạn trong thư mục workspace."""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

MAX_OUTPUT = 4000


class ToolError(Exception):
    pass


def _clip(s: str) -> str:
    return s if len(s) <= MAX_OUTPUT else s[:MAX_OUTPUT] + "\n...[đã cắt]"


class Workspace:
    def __init__(self, root: str | Path, run_timeout_s: int = 20):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_timeout_s = run_timeout_s

    def resolve(self, rel: str) -> Path:
        target = (self.root / rel).resolve()
        if target != self.root and self.root not in target.parents:
            raise ToolError(f"Đường dẫn nằm ngoài workspace: {rel}")
        return target

    def read_file(self, path: str) -> str:
        p = self.resolve(path)
        if not p.is_file():
            raise ToolError(f"Không tìm thấy file: {path}")
        return _clip(p.read_text(encoding="utf-8", errors="replace"))

    def write_file(self, path: str, content: str) -> str:
        p = self.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Đã ghi {len(content)} ký tự vào {path}"

    def list_dir(self, path: str = ".") -> str:
        p = self.resolve(path)
        if not p.is_dir():
            raise ToolError(f"Không phải thư mục: {path}")
        names = sorted(c.name + ("/" if c.is_dir() else "") for c in p.iterdir())
        return _clip("\n".join(names) or "(trống)")

    def run_python(self, code: str) -> str:
        # Lưu ý: đây KHÔNG phải sandbox bảo mật. Chỉ giới hạn thời gian và thư mục chạy.
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=self.run_timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"Quá thời gian {self.run_timeout_s}s") from exc
        out = proc.stdout
        if proc.stderr:
            out += "\n[stderr]\n" + proc.stderr
        return _clip(out or "(không có đầu ra)")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[..., str]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def build_tools(ws: Workspace) -> dict[str, Tool]:
    def obj(props: dict, required: list[str]) -> dict:
        return {"type": "object", "properties": props, "required": required}

    tools = [
        Tool("read_file", "Đọc nội dung một file trong workspace",
             obj({"path": {"type": "string"}}, ["path"]), ws.read_file),
        Tool("write_file", "Ghi nội dung vào một file trong workspace",
             obj({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
             ws.write_file),
        Tool("list_dir", "Liệt kê file và thư mục con trong workspace",
             obj({"path": {"type": "string"}}, []), ws.list_dir),
        Tool("run_python", "Chạy đoạn code Python ngắn trong workspace, có giới hạn thời gian",
             obj({"code": {"type": "string"}}, ["code"]), ws.run_python),
    ]
    return {t.name: t for t in tools}


def execute(tools: dict[str, Tool], name: str, arguments: dict) -> tuple[bool, str]:
    tool = tools.get(name)
    if tool is None:
        return False, f"Công cụ không tồn tại: {name}"
    try:
        return True, tool.run(**arguments)
    except ToolError as exc:
        return False, str(exc)
    except TypeError as exc:
        return False, f"Sai tham số: {exc}"
