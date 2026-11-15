# Chạy llama-server với model Q4_K_M và (tùy chọn) các LoRA GGUF.
# Ví dụ:
#   .\scripts\run_server.ps1
#   .\scripts\run_server.ps1 -Lora models\lora\agent.gguf
param(
    [string]$Model = "models\base-q4_k_m.gguf",
    [string[]]$Lora = @(),
    [int]$Ctx = 4096,
    [int]$Threads = 4,
    [int]$Port = 8080
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$server = "bin\llama-server.exe"
if (-not (Test-Path $server)) {
    Write-Error "Thiếu $server. Tải bản llama.cpp Windows AVX2 và giải nén vào thư mục bin\."
    exit 1
}

$serverArgs = @("-m", $Model, "-c", $Ctx, "-t", $Threads, "--host", "127.0.0.1", "--port", $Port)
foreach ($l in $Lora) { $serverArgs += @("--lora", $l) }

& $server @serverArgs
