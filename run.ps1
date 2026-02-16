#!/usr/bin/env pwsh
# ─── Posture Analysis – Local Runner (PowerShell) ─────────────────────────────
# Starts all three services (model_server, backend, frontend) locally.
# Press Ctrl+C to stop everything.

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

if (-not $env:MODEL_PORT) { $env:MODEL_PORT = "8001" }
if (-not $env:BACKEND_PORT) { $env:BACKEND_PORT = "8000" }
if (-not $env:FRONTEND_PORT) { $env:FRONTEND_PORT = "5173" }

$ModelPort = [int]$env:MODEL_PORT
$BackendPort = [int]$env:BACKEND_PORT
$FrontendPort = [int]$env:FRONTEND_PORT

$Processes = New-Object System.Collections.Generic.List[System.Diagnostics.Process]

function Test-PortListening {
    param([int]$Port)

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        $connected = $async.AsyncWaitHandle.WaitOne(500)
        if ($connected -and $client.Connected) {
            $client.EndConnect($async)
            return $true
        }
        return $false
    } finally {
        $client.Close()
    }
}

function Add-Process {
    param([System.Diagnostics.Process]$Process)

    if ($null -ne $Process) {
        $Processes.Add($Process) | Out-Null
    }
}

try {
    if (Test-PortListening -Port $ModelPort) {
        Write-Host "==> Model Server already running on http://localhost:$ModelPort"
    } else {
        Write-Host "==> Starting Model Server on http://localhost:$ModelPort ..."
        $oldPort = $env:PORT
        $env:PORT = $ModelPort
        $modelProc = Start-Process -FilePath "python" -ArgumentList "server.py" -WorkingDirectory (Join-Path $Root "model_server") -NoNewWindow -PassThru
        Add-Process -Process $modelProc
        if ($null -ne $oldPort) { $env:PORT = $oldPort } else { Remove-Item Env:PORT -ErrorAction SilentlyContinue }
    }

    if (Test-PortListening -Port $BackendPort) {
        Write-Host "==> Backend API already running on http://localhost:$BackendPort"
    } else {
        Write-Host "==> Starting Backend API on http://localhost:$BackendPort ..."
        $oldInferenceUrl = $env:INFERENCE_API_URL
        $env:INFERENCE_API_URL = "http://localhost:$ModelPort/predict"
        $backendProc = Start-Process -FilePath "uvicorn" -ArgumentList "app.main:app --host 0.0.0.0 --port $BackendPort --reload" -WorkingDirectory (Join-Path $Root "backend") -NoNewWindow -PassThru
        Add-Process -Process $backendProc
        if ($null -ne $oldInferenceUrl) { $env:INFERENCE_API_URL = $oldInferenceUrl } else { Remove-Item Env:INFERENCE_API_URL -ErrorAction SilentlyContinue }
    }

    Write-Host "==> Starting Frontend on http://localhost:$FrontendPort ..."
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npmCmd) {
        throw "npm was not found in PATH. Please install Node.js and ensure npm is available."
    }
    $npmPath = $npmCmd.Source
    $npmExt = [System.IO.Path]::GetExtension($npmPath).ToLowerInvariant()
    $frontendArgs = "run dev -- --port $FrontendPort"
    if ($npmExt -eq ".ps1") {
        $shellCmd = Get-Command pwsh -ErrorAction SilentlyContinue
        if (-not $shellCmd) {
            $shellCmd = Get-Command powershell -ErrorAction SilentlyContinue
        }
        if (-not $shellCmd) {
            throw "PowerShell host not found (pwsh or powershell)."
        }
        $frontendProc = Start-Process -FilePath $shellCmd.Source -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$npmPath`" $frontendArgs" -WorkingDirectory (Join-Path $Root "frontend") -NoNewWindow -PassThru
    } elseif ($npmExt -eq ".cmd" -or $npmExt -eq ".bat") {
        $frontendProc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "$npmPath $frontendArgs" -WorkingDirectory (Join-Path $Root "frontend") -NoNewWindow -PassThru
    } else {
        $frontendProc = Start-Process -FilePath $npmPath -ArgumentList $frontendArgs -WorkingDirectory (Join-Path $Root "frontend") -NoNewWindow -PassThru
    }
    Add-Process -Process $frontendProc

    Write-Host ""
    Write-Host "All services started:"
    Write-Host "  Frontend  -> http://localhost:$FrontendPort"
    Write-Host "  Backend   -> http://localhost:$BackendPort"
    Write-Host "  Model API -> http://localhost:$ModelPort"
    Write-Host ""
    Write-Host "Press Ctrl+C to stop all services."

    if ($Processes.Count -gt 0) {
        Wait-Process -Id ($Processes | ForEach-Object { $_.Id })
    }
} catch {
    Write-Host ""
    Write-Host "Stopping all services due to error..."
    throw
} finally {
    foreach ($proc in $Processes) {
        if (-not $proc.HasExited) {
            $proc.Kill($true)
        }
    }
}
