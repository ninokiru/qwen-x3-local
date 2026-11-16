from app.tool_calls import parse_reply, tool_prompt


def test_parses_single_call_and_text():
    content = 'Để đọc file:\n<tool_call>\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>'
    parsed = parse_reply(content)
    assert parsed.errors == []
    assert parsed.calls[0].name == "read_file"
    assert parsed.calls[0].arguments == {"path": "a.txt"}
    assert parsed.text == "Để đọc file:"


def test_reports_invalid_json():
    parsed = parse_reply("<tool_call>{name: read_file}</tool_call>")
    assert parsed.calls == []
    assert parsed.errors and "JSON" in parsed.errors[0]


def test_reports_missing_fields():
    parsed = parse_reply('<tool_call>{"arguments": {}}</tool_call>')
    assert parsed.calls == [] and parsed.errors


def test_plain_text_has_no_calls():
    parsed = parse_reply("Thủ đô là Hà Nội.")
    assert parsed.calls == [] and parsed.errors == []
    assert parsed.text == "Thủ đô là Hà Nội."


def test_tool_prompt_lists_tools():
    text = tool_prompt([{"name": "list_dir", "description": "liệt kê", "parameters": {"type": "object"}}])
    assert "list_dir" in text and "<tool_call>" in text


def test_limits_tool_call_count():
    block = '<tool_call>{"name":"list_dir","arguments":{}}</tool_call>'
    parsed = parse_reply(block * 5)
    assert len(parsed.calls) == 4
    assert parsed.errors
