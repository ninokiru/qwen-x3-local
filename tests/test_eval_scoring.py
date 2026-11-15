from evals.tool_call_eval import score_case

CALL = '<tool_call>\n{"name": "read_file", "arguments": {"path": "config.yaml"}}\n</tool_call>'


def test_correct_call_scores_full():
    r = score_case(CALL, "read_file", {"path": "config.yaml"})
    assert r == {"format_ok": True, "tool_ok": True, "args_ok": True}


def test_wrong_args_partial_score():
    r = score_case(CALL, "read_file", {"path": "other.yaml"})
    assert r["format_ok"] and r["tool_ok"] and not r["args_ok"]


def test_wrong_tool_name():
    r = score_case(CALL, "list_dir", {})
    assert r["format_ok"] and not r["tool_ok"]


def test_no_call_expected_and_given():
    assert score_case("Thủ đô là Hà Nội.", None, None)["tool_ok"] is True
    assert score_case(CALL, None, None)["tool_ok"] is False
