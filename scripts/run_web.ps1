# Chạy Web UI local; chỉ bind 127.0.0.1 theo cấu hình mặc định.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = ".\.venv\Scripts\python.exe"
if (Test-Path $python) {
    & $python -m app --config configs\app.json --web
} else {
    py -3.11 -m app --config configs\app.json --web
}
