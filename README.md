# Qwen X3 Local v1.0

Trợ lý AI cá nhân chạy local trên Windows 11, định hướng 8GB RAM và offline-first. Runtime lõi dùng thư viện chuẩn Python; model được phục vụ bằng `llama.cpp`. Model mặc định của dự án vẫn là Qwen2.5-3B-Instruct Q4_K_M theo gói model do người vận hành tự chuẩn bị.

## Có gì trong bản này

- **Chat CLI + Web UI:** Web UI dùng `http.server`, không cần Node.js, React, FastAPI hay database server.
- **Agent:** giới hạn số bước, lịch sử có giới hạn, phân loại intent nhẹ và tool call dạng JSON thống nhất.
- **Coding assistant:** quét cấu trúc project, tìm chuỗi trong source, xem trước unified diff, đọc project context và chạy pytest khi được duyệt.
- **Tool permissions:** tool ghi file, chạy Python và chạy test yêu cầu xác nhận ở CLI. Web UI chỉ được dùng tool đọc.
- **Memory:** SQLite, FTS5, chuẩn hóa dấu tiếng Việt, ghi nhớ theo loại, xóa ghi nhớ và lưu lịch sử hội thoại.
- **RAG:** lập chỉ mục tài liệu thành chunk và truy xuất qua FTS5; TXT/MD/source code dùng ngay, DOCX dùng thư viện chuẩn, PDF cần extra `pypdf`.
- **Model management:** kiểm tra model/LoRA local và tạo đối số llama-server; không tự tải model, không giả định hot-swap LoRA.
- **Voice tùy chọn:** adapter Vosk STT từ WAV local và pyttsx3 TTS; cần cài extra và tự chuẩn bị model Vosk.

## Cấu trúc

```text
app/
  agent.py, router.py, llm_client.py, model_manager.py
  tools.py, tool_calls.py, memory.py, planning.py, summarizer.py
  web.py, voice.py, health.py
  rag/       loader, chunker, indexer/retriever
  coding/    project scanner, code search, patch preview
configs/     app.json (cấu hình 8GB RAM)
data/        dữ liệu train, database memory tạo khi chạy
train/       quy trình LoRA trên Colab
models/      model GGUF và LoRA do người dùng cung cấp
bin/         llama-server.exe do người dùng cung cấp
workspace/  vùng file agent được phép thao tác
scripts/     cài đặt, chạy model, chạy Web UI, benchmark
```

## Cài đặt trên Windows 11

1. Cài Python 3.11 và chuẩn bị `bin\llama-server.exe` bản phù hợp CPU.
2. Đặt file model GGUF vào `models\Qwen2.5-3B-Instruct-Abliterated.Q4_K_M.gguf` (đúng tên file đang có trong thư mục `models`).
3. Trong PowerShell tại thư mục dự án:

```powershell
.\scripts\setup_windows.ps1
.\scripts\run_server.ps1
```

4. Mở cửa sổ PowerShell thứ hai:

```powershell
.\.venv\Scripts\python.exe -m app --config configs\app.json
```

5. Hoặc mở Web UI:

```powershell
.\scripts\run_web.ps1
```

Mặc định Web UI ở `http://127.0.0.1:8765`. Nó chỉ bind loopback, không dành để public ra Internet.

Mặc định bước setup chạy bộ test nhanh (không chạy `tests/test_generate_dataset.py`, vì bộ này thực thi nhiều mẫu code bằng subprocess). Để chạy cả bộ test đầy đủ sau khi cài đặt, dùng:

```powershell
.\scripts\setup_windows.ps1 -FullTests
```

### Chạy hoàn toàn offline

Runtime không cần truy cập Internet. Để cài package tùy chọn khi offline, chuẩn bị wheel trước trên máy có mạng:

```powershell
.\.venv\Scripts\python.exe -m pip download -d wheelhouse "pytest>=8" "pypdf>=5" "setuptools>=68" wheel
```

Chép thư mục `wheelhouse` sang máy offline rồi chạy:

```powershell
.\scripts\setup_windows.ps1 -Offline
```

Các file model GGUF, LoRA, Vosk và wheel không được đóng gói trong source ZIP; cần tự cung cấp trước khi chạy.

## Cấu hình 8GB RAM

`configs/app.json` đặt mục tiêu `context_size=4096`, `threads=2`, `max_tokens=768`, lịch sử agent tối đa 12 message, một model server và một yêu cầu xử lý tại một thời điểm trong CLI. Đây là giá trị khởi đầu, không phải cam kết model luôn vừa RAM. Nếu thiếu RAM, giảm context xuống 2048, max_tokens xuống 512 và tắt ứng dụng nền.

`runtime.offline_mode` là cấu hình chính sách ứng dụng, không phải tường lửa hệ điều hành. Đặc biệt, nếu bật thực thi Python, mã Python vẫn có quyền mạng và filesystem theo user Windows. Chỉ bật `allow_unsafe_python` khi bạn hiểu rủi ro.

## Lệnh CLI

- `/help`: xem lệnh.
- `/add <nội dung>`: lưu ghi nhớ lâu dài.
- `/memories`: liệt kê ghi nhớ.
- `/forget <id>`: xóa một ghi nhớ.
- `/search <từ khóa>`: tìm trong memory và tài liệu đã index.
- `/reset`: tạo context hội thoại mới.
- `/web`: mở Web UI.
- `/exit`: thoát.

## RAG và định dạng tài liệu

Tải file trong Web UI. File được lưu tại `workspace/documents/` và chia chunk vào SQLite FTS5. Hỗ trợ TXT, Markdown, source code, JSON, YAML, TOML, DOCX. PDF cần cài:

```powershell
.\.venv\Scripts\python.exe -m pip install "pypdf>=5"
```

Tìm kiếm hiện tại là full-text/keyword có chuẩn hóa dấu tiếng Việt, **chưa phải semantic embedding search**. Đây là lựa chọn có chủ đích để giữ footprint thấp và offline mặc định trên máy 8GB RAM.

## LoRA / model

- LoRA được nạp lúc khởi động bằng tham số `--lora`.
- Không tự động đổi LoRA giữa chat/code/agent trong cùng một server vì cách hot-swap phụ thuộc phiên bản llama.cpp.
- Để chuyển adapter, dừng server và chạy lại với adapter tương ứng.
- Không khởi chạy nhiều server/model đồng thời trên máy 8GB trừ khi đã đo RAM thực tế.

## Voice offline (tùy chọn)

Cài extra:

```powershell
.\.venv\Scripts\python.exe -m pip install ".[voice]"
```

`app.voice.transcribe_wav()` yêu cầu WAV mono PCM 16-bit và thư mục model Vosk đã có trên máy. `app.voice.speak_local()` dùng TTS local của hệ điều hành thông qua pyttsx3. Voice chưa được nối vào Web UI/CLI mặc định để tránh tự bật microphone hoặc nạp model âm thanh nền.

## Train và đánh giá

Quy trình train LoRA cũ vẫn nằm trong `train/colab_train.py`; train trên Colab, không train trên máy 8GB. Dữ liệu sinh theo template chỉ nên là dữ liệu bổ trợ. Chạy:

```powershell
.\.venv\Scripts\python.exe -m data.generate_dataset --out-dir data\generated --scale 1.4
.\.venv\Scripts\python.exe -m data.prepare_dataset --inputs data\raw\*.jsonl data\generated\*.jsonl --out-dir data\processed
.\.venv\Scripts\python.exe -m evals.run_all --label baseline
```

Cần kiểm tra thủ công model repo, phiên bản Unsloth và quá trình export GGUF trên Colab; source hiện không chứa weights hoặc model đã train.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Các test không cần model thật, GPU hoặc Internet. Một số test generator có thể chạy lâu hơn test đơn vị thông thường.

## Giới hạn cần biết

- Chưa chạy xác nhận trên Windows 11 và Pentium Gold G7400 trong môi trường thực tế của người dùng; benchmark RAM/tốc độ phải chạy trên máy đích.
- Không có sandbox OS thật cho Python. Mặc định tool này bị tắt; mỗi lần thực thi cần duyệt ở CLI.
- Web UI chỉ được dùng các tool đọc; thao tác ghi file/chạy mã cần CLI phê duyệt.
- RAG hiện dùng FTS5, chưa có embedding ngữ nghĩa, reranker neural hoặc OCR PDF scan.
- Voice là API tùy chọn, chưa có giao diện ghi âm trực tiếp.
- Router và planner vẫn là heuristic, không phải model planner độc lập.
- Không có LoRA hot-swap đảm bảo cho mọi phiên bản llama.cpp.
- Dataset và eval có sẵn là bản nháp, cần mở rộng bằng dữ liệu thực tế trước fine-tune.
- Model abliterated có thể hành xử không an toàn; luôn giữ quyền xác nhận thao tác và kiểm tra đầu ra.

Chi tiết kiến trúc: `docs/ARCHITECTURE.md`. Ghi chú nâng cấp: `docs/UPGRADE_NOTES_v1.0.md`.

## Phạm vi và giới hạn v1.0

Bản v1.0 ưu tiên ổn định và footprint nhỏ, nên một số năng lực nâng cao chưa bật hoặc chưa tích hợp hoàn chỉnh:

- RAG và memory hiện dùng SQLite FTS5/keyword search; semantic embedding và neural reranker chưa được triển khai.
- Agent planner hiện là heuristic nhẹ kết hợp vòng lặp tool có giới hạn; chưa phải hệ thống lập kế hoạch đa tác nhân.
- Web UI trả lời theo request/response, chưa có token streaming và chưa có đăng nhập; chỉ dùng trên localhost.
- Voice chỉ cung cấp adapter WAV/Vosk và TTS local; chưa tích hợp microphone hoặc nút voice vào UI.
- Python execution mặc định bị tắt. Approval callback không phải sandbox hệ điều hành.
- LoRA được nạp khi khởi động server; không có hot-swap adapter trong lúc server đang chạy.

Các mục trên được để ngoài mặc định nhằm tránh tăng RAM, phụ thuộc và bề mặt tấn công trên máy 8GB. Có thể triển khai riêng sau khi đo hiệu năng trên máy đích.
