from evals.chat_eval import score_case


def test_topic_and_vietnamese_pass():
    r = score_case("Thủ đô của Việt Nam là Hà Nội.", ["Hà Nội"])
    assert r == {"has_topics": True, "no_tool": True, "in_vietnamese": True}


def test_missing_topic_fails():
    r = score_case("Thủ đô của Việt Nam là thành phố lớn nhất phía bắc.", ["Hà Nội"])
    assert not r["has_topics"] and r["in_vietnamese"]


def test_ascii_only_answer_not_vietnamese():
    r = score_case("It is Hanoi, the capital city.", ["Hanoi"])
    assert r["has_topics"] and not r["in_vietnamese"]


def test_tool_call_in_chat_mode_fails_no_tool():
    r = score_case('Hà Nội.\n<tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>', ["Hà Nội"])
    assert not r["no_tool"]
