"""Kiểm tra dữ liệu tool_call khớp với runtime v1.0: prompt, tên tool và tham số."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tool_calls import parse_reply
from app.tools import Workspace, _validate_arguments, build_tools
from data.tool_system_prompt import TOOL_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
# Mẫu học cách sửa lỗi: lượt gọi sai tham số được phép, miễn là tool trả về lỗi như vậy.
_ERROR_PREFIXES = ("Thiếu tham số", "Tham số không", "Công cụ không tồn tại")
TRAIN_FILES = sorted((ROOT / "data" / "raw").glob("tool_call*.jsonl"))
EVAL_FILES = sorted((ROOT / "evals" / "cases").glob("tool_call*.jsonl"))


def _load(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                yield path.name, line_no, json.loads(line)


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    return build_tools(Workspace(tmp_path_factory.mktemp("ws")))


def test_fixture_files_exist():
    assert TRAIN_FILES, "không tìm thấy data/raw/tool_call*.jsonl"
    assert EVAL_FILES, "không tìm thấy evals/cases/tool_call*.jsonl"


@pytest.mark.parametrize("path", TRAIN_FILES, ids=lambda p: p.name)
def test_training_system_prompt_matches_runtime(path):
    for name, line_no, rec in _load(path):
        system = [m for m in rec["messages"] if m["role"] == "system"]
        assert len(system) == 1, f"{name}:{line_no} phải có đúng một system message"
        assert system[0]["content"] == TOOL_SYSTEM_PROMPT, f"{name}:{line_no} khác system prompt runtime"


@pytest.mark.parametrize("path", TRAIN_FILES, ids=lambda p: p.name)
def test_training_tool_calls_use_known_tools_and_valid_args(path, registry):
    for name, line_no, rec in _load(path):
        msgs = rec["messages"]
        for idx, msg in enumerate(msgs):
            if msg["role"] != "assistant":
                continue
            nxt = msgs[idx + 1] if idx + 1 < len(msgs) else {}
            expects_error = nxt.get("role") == "tool" and str(nxt.get("content", "")).startswith(_ERROR_PREFIXES)
            for call in parse_reply(msg["content"]).calls:
                if expects_error:
                    continue
                where = f"{name}:{line_no}"
                assert call.name in registry, f"{where} tool không tồn tại: {call.name}"
                issue = _validate_arguments(registry[call.name], call.arguments)
                assert issue is None, f"{where} {call.name}: {issue}"


@pytest.mark.parametrize("path", EVAL_FILES, ids=lambda p: p.name)
def test_eval_expected_tools_exist(path, registry):
    for name, line_no, case in _load(path):
        expected = case.get("expect_tool")
        if expected is None:
            continue
        assert expected in registry, f"{name}:{line_no} expect_tool không tồn tại: {expected}"
        if case.get("expect_args"):
            # expect_args chỉ kiểm tra các khóa đã khai báo, nên không ép đủ tham số bắt buộc.
            unknown = set(case["expect_args"]) - set(registry[expected].parameters["properties"])
            assert not unknown, f"{name}:{line_no} expect_args có khóa lạ: {sorted(unknown)}"
