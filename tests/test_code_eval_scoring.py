from evals.code_eval import score_case

CODE_BLOCK = "```python\ndef f(x):\n    return x * 2\n```\nHàm nhân đôi giá trị truyền vào."


def test_code_block_with_keywords_scores_full():
    r = score_case(CODE_BLOCK, ["def", "return"])
    assert r == {"has_code": True, "has_keywords": True, "no_tool": True}


def test_text_only_fails_has_code():
    r = score_case("Trong Python, từ khóa def dùng để khai báo hàm.", ["def"])
    assert not r["has_code"] and r["has_keywords"] and r["no_tool"]


def test_missing_keyword_fails():
    r = score_case(CODE_BLOCK, ["while"])
    assert r["has_code"] and not r["has_keywords"]


def test_two_blocks_required():
    assert not score_case(CODE_BLOCK, ["def"], min_code_blocks=2)["has_code"]
    two = CODE_BLOCK + "\n```python\nprint(f(3))\n```"
    assert score_case(two, ["def"], min_code_blocks=2)["has_code"]


def test_tool_call_in_code_mode_fails_no_tool():
    r = score_case(CODE_BLOCK + '\n<tool_call>{"name": "run_python", "arguments": {"code": "1"}}</tool_call>', ["def"])
    assert not r["no_tool"]
