#!/usr/bin/env pwsh
# ─── One-time setup: install all dependencies (PowerShell) ────────────────────
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

Write-Host "==> Installing backend dependencies..."
Set-Location (Join-Path $Root "backend")
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "==> Installing model server dependencies..."
Set-Location (Join-Path $Root "model_server")
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "==> Installing frontend dependencies..."
Set-Location (Join-Path $Root "frontend")
npm install

Write-Host ""
Write-Host "Setup complete! Run: pwsh ./run.sh"
