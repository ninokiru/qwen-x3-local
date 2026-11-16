from app.coding.project_scanner import scan_project, project_context
from app.coding.code_search import search_code
from app.coding.patcher import make_unified_diff


def test_scan_and_search_project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def hello():\n    pass\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "skip.py").write_text("hello", encoding="utf-8")
    assert "src/main.py" in scan_project(tmp_path)
    assert not any("skip.py" in x for x in scan_project(tmp_path))
    assert "src/main.py:1" in search_code(tmp_path, "hello")[0]
    assert "def hello" in project_context(tmp_path)


def test_diff_preview():
    diff = make_unified_diff("a.py", "x = 1\n", "x = 2\n")
    assert "-x = 1" in diff and "+x = 2" in diff
