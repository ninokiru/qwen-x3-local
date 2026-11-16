# Hướng dẫn bảo mật Qwen X3 Local

## Mặc định

- llama-server và Web UI chỉ bind `127.0.0.1`.
- Tool `run_python` bị tắt (`allow_unsafe_python=false`).
- Web UI chỉ được đăng ký tool đọc.
- `write_file`, `run_python`, `run_tests` yêu cầu callback phê duyệt ở CLI.
- Tất cả đường dẫn tool được resolve và giới hạn trong `workspace/`.
- Upload chỉ chấp nhận extension trong allowlist và giới hạn kích thước.

## Lưu ý quan trọng

**Yêu cầu phê duyệt không đồng nghĩa với sandbox.** Nếu bật `allow_unsafe_python`, code được chạy bằng tài khoản Windows hiện tại. Code đó có thể truy cập file mà user có quyền, tạo tiến trình và sử dụng mạng. Hãy chỉ chạy code bạn tin cậy.

Để bảo vệ máy khi cần thực thi code không đáng tin, chạy Qwen X3 bên trong Windows Sandbox/VM hoặc tài khoản Windows riêng quyền thấp, tắt mạng ở tầng hệ điều hành và chỉ mount một workspace dùng một lần. Cách này nằm ngoài phạm vi runtime Python hiện tại.

## Web UI

Không thay `web.host` thành `0.0.0.0` trên mạng công cộng. Web UI chưa có đăng nhập, CSRF token hay TLS; chỉ dùng trên máy local. Người có quyền truy cập localhost có thể đọc tài liệu đã upload và ghi nhớ trong SQLite.

## Dữ liệu

`data/qwen_x3.sqlite3` và `workspace/documents/` chứa dữ liệu người dùng; không đưa vào repo công khai. Backup database trước khi nâng cấp. Không upload file `.env`, token, mật khẩu hoặc private key vào RAG.
