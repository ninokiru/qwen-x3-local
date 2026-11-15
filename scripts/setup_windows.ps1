# Tạo môi trường Python và cài công cụ kiểm thử (runtime không cần thư viện ngoài).
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
