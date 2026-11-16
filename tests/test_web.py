import base64
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from app.agent import Agent
from app.config import AppConfig
from app.memory import Memory
from app.tools import Workspace, build_tools
from app.web import make_handler


class FakeClient:
    def chat(self, messages):
        return "Xin chào từ model giả lập."

    def health(self):
        return True


def test_web_health_chat_and_document_upload(tmp_path):
    cfg = AppConfig()
    cfg.workspace = "workspace"
    cfg.web.max_upload_mb = 1
    memory = Memory(tmp_path / "memory.sqlite3")
    agent = Agent(FakeClient(), build_tools(Workspace(tmp_path / "workspace")))
    handler = make_handler(agent, memory, cfg, __import__("app.rag", fromlist=["DocumentIndexer"]).DocumentIndexer(memory), tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/health") as response:
            assert json.loads(response.read())["model"] is True
        request = urllib.request.Request(base + "/api/chat", data=json.dumps({"message": "Xin chào"}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            chat_result = json.loads(response.read())
            assert "Xin chào" in chat_result["answer"]
            conversation_id = chat_result["conversation_id"]
        with urllib.request.urlopen(base + f"/api/messages?id={conversation_id}") as response:
            saved = json.loads(response.read())["messages"]
            assert [item["role"] for item in saved] == ["user", "assistant"]
        content = b"Qwen X3 local docs"
        body = {"name": "guide.txt", "data": base64.b64encode(content).decode()}
        request = urllib.request.Request(base + "/api/upload", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read())
            assert result["chunks"] == 1
        assert "Qwen X3" in " ".join(t for _, t in memory.search("Qwen X3"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        memory.close()
