# Kiến trúc v0.3

## Nguyên tắc

1. **Model đông lạnh, tùy biến bằng LoRA.** Trọng số gốc Q4_K_M không đổi. Chuyên môn được thêm qua adapter LoRA.
2. **Train ở đâu, chạy ở đâu tách biệt.** Train cần GPU nên làm trên Colab. Máy local chỉ suy luận và đánh giá.
3. **Phụ thuộc tối thiểu.** Runtime dùng thư viện chuẩn Python. Không cần PyTorch trên máy local.
4. **Định dạng tool call thống nhất.** Dữ liệu train, prompt hệ thống, parser và eval đều dùng cùng khối `<tool_call>{...}</tool_call>`. Nhờ vậy không cần phụ thuộc cờ đặc biệt của server.

## Luồng xử lý một yêu cầu

```
user → memory.search (thêm ngữ cảnh) → router.route (chat/code/agent)
     → agent.ask:
           system prompt (+ mô tả công cụ nếu mode = agent)
           → llama-server → parse_reply
              ├─ có tool_call → execute trong workspace → thêm role "tool" → lặp lại
              ├─ lỗi định dạng → nhắc model sửa → lặp lại
              └─ không có tool_call → trả lời
```

`max_steps` giới hạn số vòng gọi công cụ, tránh lặp vô hạn.

## Các thành phần

| Module | Vai trò |
|---|---|
| `app/llm_client.py` | Gọi `/v1/chat/completions`, chỉ dùng `urllib` |
| `app/tool_calls.py` | Phân tích khối tool_call và sinh mô tả công cụ |
| `app/tools.py` | Công cụ và kiểm soát đường dẫn trong workspace |
| `app/agent.py` | Vòng lặp agent |
| `app/router.py` | Chọn chế độ và ánh xạ sang LoRA |
| `app/memory.py` | SQLite FTS5 với `remove_diacritics` |
| `data/prepare_dataset.py` | Kiểm tra định dạng, thống kê tỷ lệ tiếng Việt và tool call, chia train/eval |
| `evals/tool_call_eval.py` | Chấm format, tên công cụ, tham số |
| `evals/code_eval.py` | Chấm khối code, từ khóa bắt buộc, không lẫn tool_call |
| `evals/chat_eval.py` | Chấm chủ đề cần nêu, không lẫn tool_call, trả lời tiếng Việt |
| `evals/run_all.py` | Chạy cả 3 suite, lưu báo cáo JSON vào `evals/reports/` |

## Hot-swap LoRA

Mục tiêu là một base và nhiều LoRA. Cách thực hiện phụ thuộc phiên bản llama.cpp:

- Nạp nhiều LoRA lúc khởi động bằng `--lora` (xem `scripts/run_server.ps1`).
- Đổi LoRA đang dùng khi chạy: kiểm tra endpoint điều chỉnh scale LoRA trong phiên bản đang dùng.

Nếu bản llama.cpp không hỗ trợ tốt, phương án dự phòng là hai tiến trình server, mỗi tiến trình một LoRA. Chi phí RAM sẽ tăng, nên cần đo trước.

## Ngân sách RAM (ước tính)

| Mục | Dung lượng |
|---|---|
| Base Q4_K_M | ~2 GB |
| KV cache (4096 ngữ cảnh) | ~0.15 GB |
| Mỗi LoRA | vài chục đến vài trăm MB |
| Ứng dụng Python | < 100 MB |
| Windows và phần còn lại | phần còn lại trong 8 GB |

## Việc chưa làm

- Bộ dữ liệu train thật (tiếng Việt, code, tool call).
- Embedding ngữ nghĩa cho bộ nhớ.
- Router học được, thay cho từ khóa.
- Sandbox bảo mật cho `run_python`.

## Dataset v0.4

| File | Loại | Số mẫu |
|---|---|---|
| `data/raw/tool_call.jsonl` | Agent gọi công cụ | 36 |
| `data/raw/code_vi.jsonl` | Code + giải thích tiếng Việt | 27 |
| `data/raw/chat_vi.jsonl` | Chat kiến thức tiếng Việt | 25 |
| **Tổng** | | **88 mẫu** |

Dùng `data/prepare_dataset.py` để kiểm tra, merge và chia train/eval trước khi upload Colab.
Các file trong `data/raw/` được commit; `data/processed/` và `data/generated/` thì không.

## Dataset sinh tự động (v0.5)

`data/generate_dataset.py` sinh thêm mẫu theo khuôn (template + tổ hợp ngẫu nhiên có seed).
Bậc thang `--scale` nhân số mẫu; mặc định 1.0 khoảng 1115 mẫu, `--scale 1.4` khoảng 1561 mẫu.

| File | Loại | Số mẫu (scale 1.4) | Đặc điểm |
|---|---|---|---|
| `data/generated/chat_vi.jsonl` | Hỏi đáp, tính toán, so sánh | 728 | Đáp án tính bằng Python lúc sinh |
| `data/generated/code_vi.jsonl` | Code + giải thích | 567 | Mỗi đoạn code được chạy thật, số trong giải thích phải khớp stdout |
| `data/generated/tool_call.jsonl` | Gọi công cụ, 1-nhiều bước, lỗi công cụ | 266 | Đúng định dạng `<tool_call>`, đường dẫn lỗi thật từ `Workspace.resolve` |

Quy trình: sinh theo khuôn → khử trùng lặp trong khuôn → khử câu hỏi đã có trong `data/raw` →
kiểm chứng code bằng subprocess → ghi JSONL. Nếu còn mẫu sai, generator dừng với lỗi thay vì ghi đè.

Giới hạn: văn phong lặp theo khuôn và câu hỏi có thể trùng giữa các biến thể (chat còn khoảng
77% câu hỏi duy nhất, tool_call khoảng 90%). Dùng để luyện định dạng và tiếng Việt cơ bản,
không thay được dữ liệu thật.

## Eval suite v0.4

| Script | Cases | Chỉ số |
|---|---|---|
| `evals/tool_call_eval.py` | 62 | format_ok, tool_ok, args_ok |
| `evals/code_eval.py` | 20 | has_code, has_keywords, no_tool |
| `evals/chat_eval.py` | 20 | has_topics, no_tool, in_vietnamese |

Chạy toàn bộ 1 lần: `python -m evals.run_all --label baseline`
Báo cáo lưu tại `evals/reports/` theo timestamp.
