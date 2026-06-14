[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path "data\processed\features_hourly\X_train.npy")) {
    & py -3 scripts\build_features.py --granularity hourly
    if ($LASTEXITCODE -ne 0) { throw "Build features failed" }
}

& py -3 scripts\train_lstm.py --task h1 --profile handoff
if ($LASTEXITCODE -ne 0) { throw "LSTM h1 failed" }

Write-Host "[done] LSTM h1 finished."
