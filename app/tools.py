"""Tool registry có schema, kiểm soát workspace và yêu cầu duyệt cho thao tác ghi/chạy."""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .coding.code_search import search_code as project_search_code
from .coding.project_scanner import scan_project, project_context
from .coding.patcher import preview_file_change

MAX_OUTPUT = 5000
MAX_FILE_BYTES = 1_000_000


class ToolError(Exception):
    pass


def _clip(s: str) -> str:
    return s if len(s) <= MAX_OUTPUT else s[:MAX_OUTPUT] + "\n...[đã cắt]"


class Workspace:
    def __init__(self, root: str | Path, run_timeout_s: int = 10, allow_python: bool = False):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_timeout_s = max(1, min(int(run_timeout_s), 60))
        self.allow_python = allow_python

    def resolve(self, rel: str) -> Path:
        if not isinstance(rel, str) or not rel or "\x00" in rel:
            raise ToolError("Đường dẫn không hợp lệ")
        target = (self.root / rel).resolve()
        if target != self.root and self.root not in target.parents:
            raise ToolError(f"Đường dẫn nằm ngoài workspace: {rel}")
        return target

    def read_file(self, path: str) -> str:
        p = self.resolve(path)
        if not p.is_file():
            raise ToolError(f"Không tìm thấy file: {path}")
        if p.stat().st_size > MAX_FILE_BYTES:
            raise ToolError(f"File vượt giới hạn đọc {MAX_FILE_BYTES} byte")
        return _clip(p.read_text(encoding="utf-8", errors="replace"))

    def write_file(self, path: str, content: str, overwrite: bool = False) -> str:
        p = self.resolve(path)
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ToolError(f"Nội dung vượt giới hạn ghi {MAX_FILE_BYTES} byte")
        if p.exists() and not overwrite:
            raise ToolError(f"File đã tồn tại; cần xác nhận ghi đè: {path}")
        p.parent.mkdir(parents=True, exist_ok=True)
        # Ghi nguyên tử trong cùng thư mục để tránh file dở dang nếu lỗi giữa chừng.
        temp = p.with_name(p.name + ".qwen-tmp")
        try:
            temp.write_text(content, encoding="utf-8")
            temp.replace(p)
        finally:
            if temp.exists():
                temp.unlink(missing_ok=True)
        return f"Đã ghi {len(content)} ký tự vào {path}"

    def list_dir(self, path: str = ".") -> str:
        p = self.resolve(path)
        if not p.is_dir():
            raise ToolError(f"Không phải thư mục: {path}")
        names = sorted(c.name + ("/" if c.is_dir() else "") for c in p.iterdir())
        return _clip("\n".join(names) or "(trống)")

    def search_files(self, pattern: str = "*", path: str = ".") -> str:
        root = self.resolve(path)
        if not root.is_dir():
            raise ToolError(f"Không phải thư mục: {path}")
        hits: list[str] = []
        for item in root.rglob(pattern):
            try:
                resolved = item.resolve()
                if resolved == self.root or self.root in resolved.parents:
                    hits.append(resolved.relative_to(self.root).as_posix())
            except OSError:
                continue
            if len(hits) >= 200:
                break
        return _clip("\n".join(hits) or "Không tìm thấy file phù hợp")

    def search_text(self, query: str, path: str = ".") -> str:
        if not query.strip():
            raise ToolError("Từ khóa tìm kiếm không được rỗng")
        root = self.resolve(path)
        if not root.is_dir():
            raise ToolError(f"Không phải thư mục: {path}")
        matches: list[str] = []
        skipped = {".git", ".venv", "venv", "node_modules", "__pycache__"}
        for file in root.rglob("*"):
            if not file.is_file() or any(part in skipped for part in file.parts):
                continue
            try:
                resolved = file.resolve()
                if resolved != self.root and self.root not in resolved.parents:
                    continue
                if resolved.stat().st_size > 300_000:
                    continue
                content = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            for line_no, line in enumerate(content.splitlines(), 1):
                if query.casefold() in line.casefold():
                    matches.append(f"{file.relative_to(self.root).as_posix()}:{line_no}: {line.strip()[:240]}")
                if len(matches) >= 100:
                    return _clip("\n".join(matches))
        return _clip("\n".join(matches) or "Không tìm thấy nội dung phù hợp")

    def run_python(self, code: str) -> str:
        if not self.allow_python:
            raise ToolError("run_python đang tắt. Bật allow_unsafe_python trong config và duyệt từng lần chạy.")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code], cwd=self.root,
                capture_output=True, text=True, timeout=self.run_timeout_s,
                stdin=subprocess.DEVNULL, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"Quá thời gian {self.run_timeout_s}s") from exc
        out = proc.stdout
        if proc.stderr:
            out += "\n[stderr]\n" + proc.stderr
        if proc.returncode:
            out += f"\n[exit_code={proc.returncode}]"
        return _clip(out or "(không có đầu ra)")

    def run_tests(self, target: str = "tests") -> str:
        if not self.allow_python:
            raise ToolError("Chạy test đang tắt. Bật allow_unsafe_python trong config và duyệt từng lần chạy.")
        p = self.resolve(target)
        if not p.exists():
            raise ToolError(f"Không tìm thấy mục tiêu test: {target}")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", str(p)], cwd=self.root,
                capture_output=True, text=True, timeout=self.run_timeout_s,
                stdin=subprocess.DEVNULL, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"Chạy test vượt quá {self.run_timeout_s}s") from exc
        output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        output += f"\n[exit_code={proc.returncode}]"
        return _clip(output or "(không có đầu ra)")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[..., str]
    requires_approval: bool = False

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters, "requires_approval": self.requires_approval}


def build_tools(ws: Workspace) -> dict[str, Tool]:
    def obj(props: dict, required: list[str]) -> dict:
        return {"type": "object", "properties": props, "required": required,
                "additionalProperties": False}

    def project_tree() -> str:
        files = scan_project(ws.root, max_files=300)
        return _clip("\n".join(files) or "Project chưa có file")

    def search_code(query: str) -> str:
        return _clip("\n".join(project_search_code(ws.root, query)) or "Không tìm thấy code phù hợp")

    def read_project_context() -> str:
        return _clip(project_context(ws.root, max_chars=4500))

    def preview_change(path: str, new_content: str) -> str:
        target = ws.resolve(path)
        return preview_file_change(ws.root, target.relative_to(ws.root).as_posix(), new_content)

    tools = [
        Tool("project_tree", "Liệt kê cấu trúc source project (bỏ qua thư mục build/cache)",
             obj({}, []), project_tree),
        Tool("search_code", "Tìm kiếm chuỗi trong mã nguồn project", obj({"query": {"type": "string"}}, ["query"]), search_code),
        Tool("read_project_context", "Đọc context source code gọn trong workspace", obj({}, []), read_project_context),
        Tool("preview_file_change", "Tạo unified diff để xem trước thay đổi; không ghi file", obj({"path": {"type": "string"}, "new_content": {"type": "string"}}, ["path", "new_content"]), preview_change),
        Tool("read_file", "Đọc file văn bản trong workspace",
             obj({"path": {"type": "string"}}, ["path"]), ws.read_file),
        Tool("write_file", "Tạo hoặc ghi đè file trong workspace (cần duyệt)",
             obj({"path": {"type": "string"}, "content": {"type": "string"},
                  "overwrite": {"type": "boolean"}}, ["path", "content"]),
             ws.write_file, True),
        Tool("list_dir", "Liệt kê file và thư mục trong workspace",
             obj({"path": {"type": "string"}}, []), ws.list_dir),
        Tool("search_files", "Tìm file theo glob pattern trong workspace",
             obj({"pattern": {"type": "string"}, "path": {"type": "string"}}, []), ws.search_files),
        Tool("search_text", "Tìm chuỗi trong file văn bản trong workspace",
             obj({"query": {"type": "string"}, "path": {"type": "string"}}, ["query"]), ws.search_text),
        Tool("run_python", "Chạy Python cục bộ; cần bật cấu hình và duyệt từng lần",
             obj({"code": {"type": "string"}}, ["code"]), ws.run_python, True),
        Tool("run_tests", "Chạy pytest trên mục tiêu trong workspace; cần duyệt",
             obj({"target": {"type": "string"}}, []), ws.run_tests, True),
    ]
    return {t.name: t for t in tools}


def _validate_arguments(tool: Tool, arguments: dict) -> str | None:
    if not isinstance(arguments, dict):
        return "arguments phải là object"
    schema = tool.parameters
    props = schema.get("properties", {})
    missing = [x for x in schema.get("required", []) if x not in arguments]
    extra = [x for x in arguments if x not in props]
    if missing:
        return "Thiếu tham số: " + ", ".join(missing)
    if extra:
        return "Tham số không được hỗ trợ: " + ", ".join(extra)
    for key, value in arguments.items():
        expected = props[key].get("type")
        ok = (expected == "string" and isinstance(value, str)) or (
            expected == "boolean" and isinstance(value, bool)
        )
        if not ok:
            return f"Tham số '{key}' phải có kiểu {expected}"
    return None


def execute(tools: dict[str, Tool], name: str, arguments: dict,
            approval_callback: Callable[[Tool, dict], bool] | None = None) -> tuple[bool, str]:
    tool = tools.get(name)
    if tool is None:
        return False, f"Công cụ không tồn tại: {name}"
    issue = _validate_arguments(tool, arguments)
    if issue:
        return False, issue
    if tool.requires_approval:
        if approval_callback is None or not approval_callback(tool, arguments):
            return False, f"Thao tác '{name}' chưa được người dùng phê duyệt"
    try:
        return True, tool.run(**arguments)
    except ToolError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"Không thể thực thi {name}: {type(exc).__name__}: {exc}"
