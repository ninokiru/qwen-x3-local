import pytest

from app.tools import Workspace, build_tools, execute


def test_rejects_path_outside_workspace(tmp_path):
    ws = Workspace(tmp_path / "ws")
    ok, msg = execute(build_tools(ws), "read_file", {"path": "../secret.txt"})
    assert not ok and "ngoài workspace" in msg


def test_write_then_read_roundtrip(tmp_path):
    ws = Workspace(tmp_path / "ws")
    tools = build_tools(ws)
    ok, _ = execute(tools, "write_file", {"path": "sub/hello.txt", "content": "xin chào"})
    assert ok
    ok, out = execute(tools, "read_file", {"path": "sub/hello.txt"})
    assert ok and out == "xin chào"


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
    assert not ok and "Sai tham số" in msg


def test_run_python_captures_output(tmp_path):
    ws = Workspace(tmp_path / "ws")
    ok, out = execute(build_tools(ws), "run_python", {"code": "print(sum(range(101)))"})
    assert ok and out.strip() == "5050"


def test_run_python_timeout(tmp_path):
    ws = Workspace(tmp_path / "ws", run_timeout_s=1)
    ok, msg = execute(build_tools(ws), "run_python", {"code": "while True: pass"})
    assert not ok and "Quá thời gian" in msg
