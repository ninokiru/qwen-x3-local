# Qwen X3 Local (v0.3)

Trợ lý AI chạy local trên Windows 11, cấu hình Pentium Gold G7400 / 8 GB RAM.
Model: Qwen2.5-3B-Instruct (abliterated), lượng tử hóa Q4_K_M, chạy bằng llama.cpp.
Fine-tune LoRA trên Google Colab. Ưu tiên tiếng Việt, code, và gọi công cụ.

Bản PyTorch M2 cũ đã được chuyển vào `legacy/m2-pytorch/` và không còn dùng.

## Kiến trúc

```
Colab (GPU T4)                         PC Windows 11 (local)
 dữ liệu jsonl ──► QLoRA (agent, code)─► LoRA GGUF ──┐
 bản safetensors ─► merge + Q4_K_M ───► base GGUF ───┴─► llama-server :8080
                                                            ▲
                              app/ (Python, không cần thư viện ngoài)
                              - router: chat / code / agent
                              - agent: vòng lặp tool_call
                              - tools: read/write/list/run_python (trong workspace)
                              - memory: SQLite FTS5, hỗ trợ không dấu
```

Chi tiết xem `docs/ARCHITECTURE.md`.

## Cấu trúc thư mục

```
app/          orchestrator: agent, router, tools, memory, client llama-server
data/         prepare_dataset.py: kiểm tra và chia train/eval
train/        colab_train.py: notebook train LoRA (dán từng cell)
evals/        tool_call_eval.py + cases/ : đo khả năng gọi công cụ
scripts/      run_server.ps1, bench.ps1, setup_windows.ps1
configs/      app.json
tests/        pytest (29 test, không cần model)
examples/     dữ liệu mẫu đúng định dạng
docs/         kiến trúc, quy trình
legacy/       bản M2 PyTorch cũ (tham khảo)
models/       (không commit) base-q4_k_m.gguf, lora/*.gguf
bin/          (không commit) llama-server.exe, llama-bench.exe
workspace/    thư mục mà công cụ được phép đọc/ghi
```

## Cài đặt trên Windows

1. Cài Python 3.11 (`py -3.11`).
2. Tải bản llama.cpp Windows AVX2 từ trang release của llama.cpp, giải nén `llama-server.exe` và `llama-bench.exe` vào `bin\`.
3. Đặt file model `base-q4_k_m.gguf` vào `models\`.
4. Chạy:

```powershell
.\scripts\setup_windows.ps1        # tạo .venv, cài pytest, chạy test
.\scripts\run_server.ps1           # khởi động llama-server (cổng 8080)
```

5. Ở cửa sổ khác:

```powershell
.\.venv\Scripts\python.exe -m app --config configs\app.json
```

Lệnh trong CLI: `/reset` xóa hội thoại, `/add <nội dung>` lưu ghi chú vào bộ nhớ, `/exit` thoát.

## Quy trình train LoRA trên Colab

1. Chuẩn bị dữ liệu trên PC:
   ```powershell
   .\.venv\Scripts\python.exe -m data.prepare_dataset --inputs data\raw\*.jsonl --out-dir data\processed
   ```
   Xem `examples/sample_train.jsonl` để biết định dạng. File `rejects.jsonl` liệt kê dòng lỗi.
2. Upload `data/processed/train.jsonl` lên Google Drive, tại `MyDrive/qwen-x3/data/`.
3. Mở Colab, chọn runtime GPU T4, dán từng cell từ `train/colab_train.py`.
   Đặt `MODEL` là repo safetensors của bản abliterated và `ADAPTER_NAME` là `agent` hoặc `code`.
4. Tải LoRA hoặc GGUF từ Drive về `models\`.
5. Đo chất lượng trước khi dùng:
   ```powershell
   .\.venv\Scripts\python.exe -m evals.tool_call_eval --base-url http://127.0.0.1:8080
   ```
   Chỉ số `format_ok`, `tool_ok`, `args_ok` cho biết model gọi công cụ đúng đến đâu.

## Đo tốc độ

```powershell
.\scripts\bench.ps1
```

Kết quả thật sẽ thay thế các ước tính trong tài liệu.

## Giới hạn hiện tại

- `run_python` không phải sandbox bảo mật. Nó chỉ giới hạn thời gian và thư mục chạy. Chỉ dùng với code đáng tin.
- Router là heuristic từ khóa. Chưa có bộ phân loại học.
- Bộ nhớ dùng từ khóa (FTS5), chưa có embedding ngữ nghĩa.
- Chưa có dữ liệu train thật. Bộ `evals/cases` mới là mẫu nhỏ, cần mở rộng.
- Notebook Colab chưa được chạy thử trên GPU thật. Kiểm tra lại tên repo, phiên bản Unsloth và `save_pretrained_gguf` trước khi train đầy đủ.
- Mô hình abliterated đã bỏ cơ chế từ chối. Trách nhiệm về nội dung đầu ra thuộc về người vận hành. Giấy phép Qwen2.5-3B là Qwen Research License, cần đọc kỹ trước khi dùng thương mại.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Các test không cần GPU hay model thật.
