[CmdletBinding()]
param(
    [switch]$WithAnalysis
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "[handoff] repo root: $RepoRoot"

function Invoke-Step {
    param(
        [string]$Title,
        [string[]]$Command
    )

    Write-Host ""
    Write-Host "[handoff] $Title"
    Write-Host "[handoff] cmd: $($Command -join ' ')"
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Title"
    }
}

if (-not (Test-Path "data\processed\features_hourly\X_train.npy")) {
    Invoke-Step -Title "Build hourly features" -Command @("py", "-3", "scripts\build_features.py", "--granularity", "hourly")
}

Invoke-Step -Title "Train LSTM (handoff profile, h1 + seq6)" -Command @("py", "-3", "scripts\train_lstm.py", "--task", "both", "--profile", "handoff")

Invoke-Step -Title "Train Transformer (runtime-safe config, h1 + seq6)" -Command @("py", "-3", "scripts\train_transformer.py", "--task", "both", "--epochs", "60", "--patience", "8", "--batch-size", "64", "--lr", "5e-4")

if ($WithAnalysis) {
    Invoke-Step -Title "Run hourly analysis" -Command @("py", "-3", "scripts\run_hourly_analysis.py")
    Invoke-Step -Title "Run cross-city XGBoost" -Command @("py", "-3", "scripts\run_cross_city_xgboost.py")
    Invoke-Step -Title "Assemble summary outputs" -Command @("py", "-3", "scripts\assemble_hourly_outputs.py")
    Invoke-Step -Title "Generate figures" -Command @("py", "-3", "scripts\generate_hourly_figures.py")
}

Write-Host ""
Write-Host "[handoff] done."
Write-Host "[handoff] deep-model outputs should now exist under outputs/hourly/"
Write-Host "[handoff] LSTM: lstm_predictions_h1.csv, lstm_predictions_seq6.csv, lstm_metrics.csv"
Write-Host "[handoff] Transformer: transformer_predictions_h1.csv, transformer_predictions_seq6.csv, transformer_metrics.csv"
