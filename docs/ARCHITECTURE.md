# Kiến trúc Qwen X3 Local v1.0

## Nguyên tắc

1. Model inference là tiến trình riêng `llama-server`; Python chỉ là orchestrator.
2. Runtime lõi dùng thư viện chuẩn Python để giảm RAM và chạy offline.
3. SQLite là nguồn lưu dữ liệu duy nhất cho memory, conversation và chỉ mục tài liệu.
4. FTS5 + chuẩn hóa Unicode là retrieval mặc định; không tải embedding model tự động.
5. Mọi thao tác ghi/chạy code cần quyền rõ ràng; Web UI chỉ dùng tool đọc.
6. Tool call vẫn dùng format huấn luyện `<tool_call>{...}</tool_call>` để không lệ thuộc format đặc biệt của server.

## Luồng

```text
CLI / Web UI
    |
    v
Router (heuristic)
    |
    +---- chat / code / agent
    |
    v
Memory + RAG retrieval (SQLite FTS5)
    |
    v
Agent (bounded steps + bounded history)
    |
    v
llama-server /v1/chat/completions
    |
    v
Tool-call parser (size/count/schema validation)
    |
    +---- read-only tools -> Workspace
    |
    +---- write/run tools -> explicit CLI approval -> Workspace
    |
    v
Final response + SQLite conversation record
```

## Modules

| Module | Trách nhiệm |
|---|---|
| `app/config.py` | Cấu hình lồng nhau, tương thích cấu hình v0.3 |
| `app/llm_client.py` | HTTP client, timeout, retry giới hạn, health check |
| `app/model_manager.py` | Validate local GGUF/LoRA paths, build server args |
| `app/router.py` | Classify chat/code/agent bằng luật nhẹ |
| `app/agent.py` | Context, tool loop, stop limit, approval gate |
| `app/tool_calls.py` | Parse format, giới hạn số lượng/kích thước call |
| `app/tools.py` | Workspace, schema, tool registry, permission |
| `app/memory.py` | SQLite, FTS5, normalized index, notes, conversations |
| `app/rag/` | Load TXT/MD/source/DOCX/PDF optional, chunk, retrieve |
| `app/coding/` | Project scan, code search, diff preview |
| `app/planning.py` | Kế hoạch quy tắc nhẹ, không model riêng |
| `app/web.py` | Web UI và localhost JSON API bằng `http.server` |
| `app/voice.py` | Optional offline Vosk STT / pyttsx3 TTS adapters |

## Bảo mật

- `Workspace.resolve()` resolve symlink và chặn đường dẫn ra ngoài workspace.
- File đọc/ghi giới hạn 1 MB; tool output giới hạn 5.000 ký tự.
- Tool schema từ chối tham số thiếu, thừa và sai kiểu.
- `write_file`, `run_python`, `run_tests` có `requires_approval`.
- `run_python` mặc định tắt; khi bật vẫn chạy với quyền của user Windows, không phải sandbox.
- Web UI chỉ đăng ký tool đọc, bind `127.0.0.1`, giới hạn upload và định dạng.
- Không ghi prompt vào HTTP access log.

## RAM

Cấu hình gợi ý ban đầu: context 4096, threads 2, max_tokens 768, agent history 12 message, một model server. KV cache và tổng RAM còn phụ thuộc build llama.cpp, model, backend, CPU và context thực tế. Đo trên máy đích; nếu thiếu RAM hạ context về 2048.

## LoRA

`MODE_TO_LORA` là metadata cho chat/code/agent, không bảo đảm adapter tự chuyển. Script nạp LoRA khi khởi động. Chuyển adapter cần restart server nếu phiên bản llama.cpp không cung cấp API runtime đã xác minh. Không chạy đồng thời nhiều server trên cấu hình 8GB.

## RAG

File được đọc theo extension allowlist, chia chunk mặc định 1800 ký tự và overlap 200. Index lưu vào FTS5 normalized text. Matching yêu cầu tất cả token (AND) để giảm kết quả trùng lỏng. PDF scan không có text layer thì chưa OCR.

## Những phần chưa tuyên bố hoàn tất

- Semantic embedding/reranker neural: chưa bật để tránh thêm model và RAM.
- Token streaming real-time: giao diện hiện hiển thị sau khi agent hoàn thành lượt.
- Voice capture UI: API nhận WAV local, chưa tự bật microphone.
- Sandbox cấp OS cho code execution: chưa được triển khai.
- Auto LoRA hot-swap và planner model-based: chưa có.
