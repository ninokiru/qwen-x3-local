$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"
python -m pytest -q
