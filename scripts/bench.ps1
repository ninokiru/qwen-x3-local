# Đo tốc độ trên máy thật (prompt processing và token generation).
param(
    [string]$Model = "models\base-q4_k_m.gguf",
    [int]$Threads = 2
)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not (Test-Path $Model)) { Write-Error "Không tìm thấy model: $Model"; exit 1 }
if (-not (Test-Path "bin\llama-bench.exe")) { Write-Error "Thiếu bin\llama-bench.exe"; exit 1 }
& "bin\llama-bench.exe" -m $Model -t $Threads -p 512 -n 128
