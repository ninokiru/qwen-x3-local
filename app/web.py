"""Web UI cực nhẹ, dùng http.server (không cần FastAPI/Node/npm)."""
from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import AppConfig
from .rag import DocumentIndexer
from .tools import Tool
from .router import classify

ALLOWED_UPLOADS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".json", ".jsonl", ".yaml",
    ".yml", ".toml", ".ini", ".ps1", ".sql", ".html", ".css", ".xml",
    ".csv", ".java", ".c", ".cpp", ".h", ".docx", ".pdf",
}

INDEX_HTML = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen X3 Local</title><style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#121a2d;--line:#26334d;--text:#e8eefc;--muted:#9aa9c4;--accent:#76a9fa}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#172642,var(--bg) 48%);color:var(--text);font:15px system-ui,Segoe UI,sans-serif;min-height:100vh}
.app{max-width:1050px;margin:auto;padding:24px;display:grid;grid-template-columns:260px 1fr;gap:18px}.side,.main{background:#111a2bdc;border:1px solid var(--line);border-radius:18px;min-height:calc(100vh - 48px)}
.side{padding:18px}.brand{font-size:21px;font-weight:750;letter-spacing:.4px}.sub{color:var(--muted);font-size:12px;margin-top:3px}.status{margin:18px 0;padding:10px;background:#0b1323;border:1px solid var(--line);border-radius:10px;color:var(--muted);font-size:12px}
.side h3{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin:22px 0 8px}.side button,.send,.upload{background:#203454;color:var(--text);border:1px solid #39547d;border-radius:10px;padding:9px 12px;cursor:pointer}.side button{width:100%;text-align:left;margin:4px 0}.main{display:flex;flex-direction:column;overflow:hidden}.top{padding:18px 22px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}.top small{color:var(--muted)}.messages{padding:22px;flex:1;overflow:auto;display:flex;flex-direction:column;gap:14px;min-height:300px}.msg{max-width:88%;padding:13px 15px;border-radius:14px;white-space:pre-wrap;line-height:1.55;overflow-wrap:anywhere}.user{align-self:flex-end;background:#213c63}.assistant{align-self:flex-start;background:#17243a;border:1px solid var(--line)}.composer{border-top:1px solid var(--line);padding:16px;display:flex;gap:10px}.composer textarea{flex:1;resize:vertical;min-height:54px;max-height:180px;background:#0b1323;color:var(--text);border:1px solid var(--line);border-radius:12px;padding:12px;font:inherit}.send{min-width:90px;background:#315d98}.foot{color:var(--muted);font-size:11px;padding:0 18px 14px}.file{display:none}.hint{color:var(--muted);font-size:12px;margin-top:10px}.mem{font-size:12px;color:var(--muted);word-break:break-word}
@media(max-width:760px){.app{grid-template-columns:1fr;padding:10px}.side{min-height:0}.main{min-height:70vh}.side h3{margin-top:12px}}
</style></head><body><div class="app"><aside class="side"><div class="brand">◈ QWEN X3</div><div class="sub">LOCAL INTELLIGENCE · OFFLINE-FIRST</div><div class="status" id="status">Đang kiểm tra model...</div>
<button onclick="newChat()">＋ Cuộc trò chuyện mới</button><h3>Tài liệu</h3><label class="upload" style="display:block;text-align:center">Tải tài liệu lên<input class="file" id="file" type="file" accept=".txt,.md,.pdf,.docx,.py,.json,.jsonl,.yaml,.yml,.toml,.js,.ts"></label><div class="hint" id="uploadStatus">TXT · MD · PDF* · DOCX · Source code</div><h3>Ghi nhớ</h3><button onclick="addMemory()">＋ Lưu ghi nhớ</button><div id="memories" class="mem"></div><div class="hint">* PDF cần cài extra pypdf. Giao diện chỉ bind localhost; thao tác ghi/chạy code cần dùng CLI để xác nhận.</div></aside>
<main class="main"><header class="top"><div><b>Trợ lý cá nhân</b><br><small>Qwen2.5 · llama.cpp · SQLite</small></div><small id="mode">CHAT</small></header><section class="messages" id="messages"><div class="msg assistant">Xin chào! Mình là Qwen X3. Bạn có thể trò chuyện, hỏi về code hoặc tải tài liệu để tra cứu local.</div></section><div class="composer"><textarea id="prompt" placeholder="Nhập yêu cầu... (Enter để gửi, Shift+Enter xuống dòng)"></textarea><button class="send" id="send" onclick="sendMessage()">Gửi</button></div><div class="foot">Câu trả lời được xử lý trên máy local. Không gửi dữ liệu ra Internet bởi giao diện này.</div></main></div>
<script>
const $=id=>document.getElementById(id);let busy=false;
function bubble(role,text){const d=document.createElement('div');d.className='msg '+role;d.textContent=text;$('messages').appendChild(d);$('messages').scrollTop=$('messages').scrollHeight;return d}
async function req(url,opts={}){const r=await fetch(url,opts);const j=await r.json();if(!r.ok)throw Error(j.error||r.statusText);return j}
async function check(){try{const x=await req('/api/health');$('status').textContent=x.model?'● llama-server listo':'● Web UI listo · falta conectar modelo';$('status').style.color=x.model?'#83d6a2':'#e8bd75'}catch(e){$('status').textContent='No se pudo consultar estado: '+e.message}}
async function sendMessage(){if(busy)return;const text=$('prompt').value.trim();if(!text)return;busy=true;$('send').disabled=true;$('prompt').value='';bubble('user',text);const pending=bubble('assistant','Đang suy nghĩ...');try{const data=await req('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});pending.textContent=data.answer;$('mode').textContent=data.mode.toUpperCase()}catch(e){pending.textContent='Lỗi: '+e.message}finally{busy=false;$('send').disabled=false;$('prompt').focus()}}
async function newChat(){ try{await req('/api/conversations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:'Web UI session'})});$('messages').replaceChildren();bubble('assistant','Cuộc trò chuyện mới đã bắt đầu.')}catch(e){alert(e.message)} }
async function addMemory(){const text=prompt('Nội dung cần lưu vào bộ nhớ:');if(!text)return;try{await req('/api/memories',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:'note',text})});loadMemories()}catch(e){alert(e.message)}}
async function loadMemories(){try{const x=await req('/api/memories');$('memories').textContent=x.items.map(i=>'• '+i.text).join('\n')||'Chưa có ghi nhớ'}catch(e){$('memories').textContent=e.message}}
$('file').addEventListener('change',async e=>{const f=e.target.files[0];if(!f)return;$('uploadStatus').textContent='Đang lập chỉ mục '+f.name+'...';try{const b=await f.arrayBuffer();let binary='';for(const c of new Uint8Array(b))binary+=String.fromCharCode(c);const encoded=btoa(binary);const x=await req('/api/upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:f.name,data:encoded})});$('uploadStatus').textContent='Đã index '+x.chunks+' đoạn từ '+f.name}catch(err){$('uploadStatus').textContent='Lỗi: '+err.message}finally{e.target.value=''}});
$('prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage()}});check();loadMemories();
</script></body></html>'''


def make_handler(agent, memory, config: AppConfig, indexer: DocumentIndexer, project_root: Path):
    lock = threading.RLock()
    # Trên Web UI chỉ bật các tool đọc. Tool ghi/chạy cần CLI và xác nhận tương tác.
    safe_names = {"read_file", "list_dir", "search_files", "search_text", "project_tree", "search_code", "read_project_context"}
    agent.tools = {k: v for k, v in agent.tools.items() if k in safe_names}
    conversation_id = memory.create_conversation("Web UI session")

    class Handler(BaseHTTPRequestHandler):
        server_version = "QwenX3Local/1.0"

        def log_message(self, fmt, *args):
            return  # Không ghi prompt người dùng vào access log.

        def _json(self, status: int, data: dict):
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            limit = max(1024, config.runtime.max_request_chars * 4 + config.web.max_upload_mb * 4 // 3 * 1024 * 1024 + 100_000)
            if length <= 0 or length > limit:
                raise ValueError("Kích thước request không hợp lệ")
            raw = self.rfile.read(length)
            obj = json.loads(raw.decode("utf-8"))
            if not isinstance(obj, dict):
                raise ValueError("JSON body phải là object")
            return obj

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                raw = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers()
                self.wfile.write(raw)
                return
            if parsed.path == "/api/health":
                model_ok = bool(agent.client.health()) if hasattr(agent.client, "health") else None
                return self._json(200, {"ok": True, "model": model_ok, "version": "1.0"})
            if parsed.path == "/api/memories":
                items = [{"id": i, "kind": k, "text": t[:240]} for i, k, t in memory.list_memories(30)]
                return self._json(200, {"items": items})
            if parsed.path == "/api/conversations":
                items = [{"id": i, "title": t, "updated_at": d} for i, t, d in memory.list_conversations()]
                return self._json(200, {"items": items, "current_id": conversation_id})
            if parsed.path == "/api/messages":
                query = parse_qs(parsed.query)
                try:
                    cid = int(query.get("id", [conversation_id])[0])
                except ValueError:
                    return self._json(400, {"error": "conversation id không hợp lệ"})
                return self._json(200, {"messages": memory.get_messages(cid)})
            return self._json(404, {"error": "Không tìm thấy endpoint"})

        def do_POST(self):
            nonlocal conversation_id
            parsed = urlparse(self.path)
            try:
                body = self._body()
                if parsed.path == "/api/chat":
                    message = str(body.get("message", "")).strip()
                    if not message or len(message) > config.runtime.max_request_chars:
                        return self._json(400, {"error": "Tin nhắn rỗng hoặc vượt giới hạn"})
                    hits = memory.search(message, k=config.memory.top_k) if config.memory.enabled else []
                    extra = "\n\n".join(f"[{s}] {t}" for s, t in hits)[:config.memory.max_context_chars]
                    with lock:
                        memory.save_message(conversation_id, "user", message)
                        answer = agent.ask(message, extra_context=extra or None)
                        memory.save_message(conversation_id, "assistant", answer)
                        return self._json(200, {"answer": answer, "mode": classify(message).mode,
                                                "conversation_id": conversation_id})
                if parsed.path == "/api/conversations":
                    with lock:
                        agent.reset()
                        conversation_id = memory.create_conversation(str(body.get("title", "Web UI session")))
                    return self._json(201, {"id": conversation_id})
                if parsed.path == "/api/memories":
                    memory_text = str(body.get("text", ""))
                    if len(memory_text) > 20000:
                        return self._json(400, {"error": "Ghi nhớ vượt giới hạn 20.000 ký tự"})
                    mid = memory.save_memory(str(body.get("kind", "note")), memory_text)
                    return self._json(201, {"id": mid})
                if parsed.path == "/api/upload":
                    name = Path(str(body.get("name", "upload.bin"))).name[:180]
                    ext = Path(name).suffix.lower()
                    if ext not in ALLOWED_UPLOADS:
                        return self._json(400, {"error": f"Định dạng {ext} chưa được hỗ trợ"})
                    data = base64.b64decode(str(body.get("data", "")), validate=True)
                    if len(data) > config.web.max_upload_mb * 1024 * 1024:
                        return self._json(413, {"error": "File vượt giới hạn upload"})
                    workspace_root = (project_root / config.workspace).resolve()
                    docs = workspace_root / "documents"
                    docs.mkdir(parents=True, exist_ok=True)
                    # Tên do server tạo, tên gốc chỉ dùng để hiển thị/chọn loader.
                    safe_name = name.replace("/", "_").replace("\\", "_")
                    destination = (docs / safe_name).resolve()
                    if docs.resolve() not in destination.parents or workspace_root not in destination.parents:
                        return self._json(400, {"error": "Đường dẫn upload không hợp lệ"})
                    destination.write_bytes(data)
                    chunks = indexer.index_bytes(name, data, source=f"documents/{safe_name}")
                    return self._json(201, {"name": safe_name, "chunks": chunks})
                return self._json(404, {"error": "Không tìm thấy endpoint"})
            except (ValueError, json.JSONDecodeError) as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:
                return self._json(500, {"error": f"Lỗi xử lý: {exc}"})

        def do_DELETE(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/memories/"):
                try:
                    mid = int(parsed.path.rsplit("/", 1)[-1])
                    return self._json(200, {"deleted": memory.delete_memory(mid)})
                except ValueError:
                    return self._json(400, {"error": "ID không hợp lệ"})
            return self._json(404, {"error": "Không tìm thấy endpoint"})

    return Handler


def serve(agent, memory, config: AppConfig, project_root: str | Path = ".") -> None:
    root = Path(project_root).resolve()
    indexer = DocumentIndexer(memory)
    original_tools = agent.tools
    handler = make_handler(agent, memory, config, indexer, root)
    server = ThreadingHTTPServer((config.web.host, config.web.port), handler)
    print(f"Qwen X3 Web UI: http://{config.web.host}:{config.web.port}")
    print("Web UI chỉ cho phép tool đọc; ghi/chạy code cần dùng CLI để xác nhận.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang dừng Web UI...")
    finally:
        server.server_close()
        agent.tools = original_tools
