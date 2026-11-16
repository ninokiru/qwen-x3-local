import pytest

from app.tools import Workspace, build_tools, execute


def approve(tool, args):
    return True


def test_rejects_path_outside_workspace(tmp_path):
    ws = Workspace(tmp_path / "ws")
    ok, msg = execute(build_tools(ws), "read_file", {"path": "../secret.txt"})
    assert not ok and "ngoài workspace" in msg


def test_write_then_read_roundtrip(tmp_path):
    ws = Workspace(tmp_path / "ws")
    tools = build_tools(ws)
    ok, _ = execute(tools, "write_file", {"path": "sub/hello.txt", "content": "xin chào"}, approve)
    assert ok
    ok, out = execute(tools, "read_file", {"path": "sub/hello.txt"})
    assert ok and out == "xin chào"


def test_write_requires_approval_and_prevents_overwrite(tmp_path):
    ws = Workspace(tmp_path / "ws")
    tools = build_tools(ws)
    ok, msg = execute(tools, "write_file", {"path": "a.txt", "content": "x"})
    assert not ok and "phê duyệt" in msg
    assert execute(tools, "write_file", {"path": "a.txt", "content": "x"}, approve)[0]
    ok, msg = execute(tools, "write_file", {"path": "a.txt", "content": "y"}, approve)
    assert not ok and "đã tồn tại" in msg


def test_list_dir_marks_subdirectories(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "d").mkdir()
    (ws.root / "f.txt").write_text("x", encoding="utf-8")
    ok, out = execute(build_tools(ws), "list_dir", {"path": "."})
    assert ok and out.splitlines() == ["d/", "f.txt"]


def test_unknown_tool_and_bad_args(tmp_path):
    tools = build_tools(Workspace(tmp_path / "ws"))
    assert execute(tools, "nope", {})[0] is False
    ok, msg = execute(tools, "read_file", {"wrong": "x"})
    assert not ok and "Thiếu tham số" in msg


def test_run_python_requires_config_and_approval(tmp_path):
    ws = Workspace(tmp_path / "ws", allow_python=True)
    tools = build_tools(ws)
    ok, msg = execute(tools, "run_python", {"code": "print(sum(range(101)))"})
    assert not ok and "phê duyệt" in msg
    ok, out = execute(tools, "run_python", {"code": "print(sum(range(101)))"}, approve)
    assert ok and out.strip() == "5050"


def test_run_python_timeout(tmp_path):
    ws = Workspace(tmp_path / "ws", run_timeout_s=1, allow_python=True)
    ok, msg = execute(build_tools(ws), "run_python", {"code": "while True: pass"}, approve)
    assert not ok and "Quá thời gian" in msg


def test_search_text_and_project_scanner(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "main.py").write_text("def hello():\n    return 'xin chao'", encoding="utf-8")
    tools = build_tools(ws)
    ok, out = execute(tools, "search_code", {"query": "hello"})
    assert ok and "main.py:1" in out
