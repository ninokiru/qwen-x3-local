param([switch]$Offline, [switch]$WithDocuments, [switch]$WithVoice, [switch]$FullTests)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
py -3.11 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
if (-not $Offline) { & $python -m pip install --upgrade pip }
$extras = "dev"
if ($WithDocuments) { $extras += ",documents" }
if ($WithVoice) { $extras += ",voice" }
if ($Offline) {
    if (-not (Test-Path "wheelhouse")) { Write-Error "Thiếu thư mục wheelhouse cho cài offline"; exit 1 }
    & $python -m pip install --no-index --find-links wheelhouse -e ".[${extras}]"
} else {
    & $python -m pip install -e ".[${extras}]"
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$pytestArgs = @("-q")
if (-not $FullTests) {
    # Bộ test generator chạy code mẫu qua nhiều tiến trình Python; không chạy mặc định khi setup.
    $pytestArgs += @("--ignore=tests/test_generate_dataset.py")
}
& $python -m pytest @pytestArgs
