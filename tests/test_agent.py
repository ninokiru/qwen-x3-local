from app.agent import Agent
from app.tools import Workspace, build_tools


class FakeClient:
    """Trả về lần lượt các phản hồi đã định sẵn, và ghi lại messages đã gửi."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def chat(self, messages):
        self.sent.append(messages)
        return self.replies.pop(0)


def test_agent_executes_tool_then_answers(tmp_path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "config.yaml").write_text("port: 8080", encoding="utf-8")
    client = FakeClient([
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "config.yaml"}}\n</tool_call>',
        "File đặt cổng là 8080.",
    ])
    agent = Agent(client, build_tools(ws))
    answer = agent.ask("Đọc file config.yaml giúp tôi")
    assert answer == "File đặt cổng là 8080."
    tool_msgs = [m for m in agent.history if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[0]["content"] == "port: 8080"


def test_agent_recovers_from_bad_format(tmp_path):
    client = FakeClient([
        "<tool_call>{bad json}</tool_call>",
        "Xin lỗi, đây là câu trả lời.",
    ])
    agent = Agent(client, build_tools(Workspace(tmp_path / "ws")))
    assert agent.ask("Viết hàm python") == "Xin lỗi, đây là câu trả lời."
    assert any("Lỗi định dạng" in m["content"] for m in agent.history if m["role"] == "user")


def test_agent_stops_at_max_steps(tmp_path):
    loop = '<tool_call>\n{"name": "list_dir", "arguments": {"path": "."}}\n</tool_call>'
    client = FakeClient([loop] * 3)
    agent = Agent(client, build_tools(Workspace(tmp_path / "ws")), max_steps=3)
    agent_msg = agent.ask("Liệt kê các file trong thư mục workspace")
    assert "giới hạn" in agent_msg
