# Đo tốc độ trên máy thật (prompt processing và token generation).
param(
    [string]$Model = "models\base-q4_k_m.gguf",
    [int]$Threads = 4
)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& "bin\llama-bench.exe" -m $Model -t $Threads -p 512 -n 128
