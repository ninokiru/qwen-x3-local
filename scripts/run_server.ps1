# Chạy llama.cpp local. Mặc định 1 model, context 4096, 2 CPU threads cho máy 8GB RAM.
# LoRA được nạp lúc khởi động; không giả định có hot-swap runtime.
param(
    [string]$Model = "models\base-q4_k_m.gguf",
    [string[]]$Lora = @(),
    [int]$Ctx = 4096,
    [int]$Threads = 2,
    [int]$Port = 8080
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$server = Join-Path $root "bin\llama-server.exe"
if (-not (Test-Path $server)) {
    Write-Error "Thiếu bin\llama-server.exe. Hãy cài bản llama.cpp phù hợp CPU/AVX2 vào bin\."
    exit 1
}
if (-not (Test-Path $Model)) {
    Write-Error "Không tìm thấy model: $Model"
    exit 1
}
foreach ($adapter in $Lora) {
    if (-not (Test-Path $adapter)) { Write-Error "Không tìm thấy LoRA: $adapter"; exit 1 }
}
if ($Ctx -lt 1024 -or $Ctx -gt 8192) { Write-Error "Ctx phải trong khoảng 1024..8192"; exit 1 }
if ($Threads -lt 1 -or $Threads -gt 8) { Write-Error "Threads phải trong khoảng 1..8"; exit 1 }

$serverArgs = @("-m", $Model, "-c", $Ctx, "-t", $Threads, "--host", "127.0.0.1", "--port", $Port)
foreach ($l in $Lora) { $serverArgs += @("--lora", $l) }
Write-Host "Starting llama-server on 127.0.0.1:$Port (ctx=$Ctx, threads=$Threads)"
& $server @serverArgs
