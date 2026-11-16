# Ghi chú nâng cấp v0.3 → v1.0

## Thay đổi chính

### Core
- `app/config.py`: cấu hình server/runtime/web/memory, vẫn đọc được các khóa config v0.3.
- `app/llm_client.py`: timeout/retry giới hạn, thông báo HTTP error, health check nhanh.
- `app/router.py`: intent confidence và routing heuristic rõ ràng.
- `app/agent.py`: lịch sử giới hạn, giới hạn bước, extra context, approval callback.
- `app/tool_calls.py`: giới hạn tối đa 4 tool call và 16 KB JSON mỗi call.
- `app/model_manager.py`: kiểm tra đường dẫn model GGUF/LoRA và tạo args.

### Tools và coding
- `app/tools.py`: workspace path validation, atomic write, giới hạn file/output, search files/text, project tree/context, diff preview, pytest tool.
- Ghi file/chạy Python/chạy test cần approval callback. Python execution mặc định tắt trong config.
- `app/coding/`: project scanner, code search và unified diff preview.

### Memory/RAG
- `app/memory.py`: notes có ID, CRUD tối thiểu, conversation/message tables, FTS5 normalized index hỗ trợ bỏ dấu tiếng Việt.
- `app/rag/`: text loader, DOCX parser built-in, PDF parser tùy chọn, chunker, indexing và retrieval.

### Interface
- `app/web.py`: Web UI tối giản, API chat/upload/memory, localhost only; web chỉ có tool đọc.
- `scripts/run_web.ps1`: chạy Web UI.
- `app/voice.py`: optional offline Vosk STT và pyttsx3 TTS adapters.

### Cấu hình
- `configs/app.json`: khởi điểm cho 8GB RAM (context 4096, threads 2, max_tokens 768).
- `scripts/run_server.ps1`: validation model/LoRA và giới hạn context/thread.
- `scripts/setup_windows.ps1`: extra dev/documents/voice và cài offline qua wheelhouse.

## Thứ tự nâng cấp đã thực hiện

1. Config, LLM client, router.
2. Workspace tools, tool schema, approval, agent loop.
3. SQLite memory và FTS5 chuẩn hóa tiếng Việt.
4. RAG indexing/retrieval.
5. Coding utilities và patch preview.
6. Web UI localhost và upload.
7. Voice adapters tùy chọn.
8. Test, README, security/architecture documentation.

## Cần xác minh thêm trên máy đích

- `llama-server.exe` version/build, AVX2 support, tham số CLI và adapter LoRA tương ứng.
- Model Q4_K_M và LoRA file thực tế.
- RAM peak, tốc độ token/s, nhiệt độ CPU và ổn định khi context 2048/4096.
- PDF extraction trên tập tài liệu thật.
- Vosk model/ngôn ngữ và voice playback trên Windows.
- Full eval và quality benchmark trên model local.

## Giới hạn có chủ đích trong v1.0

- Semantic embeddings/reranker chưa triển khai; retrieval dùng SQLite FTS5.
- Planner là heuristic nhẹ, không phải neural planner hoặc multi-agent orchestration.
- Web UI chưa có token streaming, authentication, CSRF token hoặc TLS; chỉ bind localhost.
- Voice adapter chưa được tích hợp vào UI/CLI và không tự mở microphone.
- `run_python` không phải sandbox; mặc định tắt và nếu bật thì chạy với quyền user Windows hiện tại.
- LoRA chỉ được nạp lúc khởi động; không có hot-swap trong runtime.
