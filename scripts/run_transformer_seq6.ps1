[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path "data\processed\features_hourly\X_train.npy")) {
    & py -3 scripts\build_features.py --granularity hourly
    if ($LASTEXITCODE -ne 0) { throw "Build features failed" }
}

& py -3 scripts\train_transformer.py --task seq6 --epochs 80 --patience 8 --batch-size 64 --lr 5e-4
if ($LASTEXITCODE -ne 0) { throw "Transformer seq6 failed" }

Write-Host "[done] Transformer seq6 finished."
