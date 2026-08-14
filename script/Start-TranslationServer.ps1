<#
.SYNOPSIS
    Start the M2M100 GPU translation server with a live status panel.

.DESCRIPTION
    Launches the translation server using the existing 'dp' Conda
    environment and shows live status: port, process state (starting /
    running / stopped), model readiness, device, and uptime. Assumes the
    environment is already initialized (run .\script\Initialize-Env.ps1
    separately when needed). Server output is streamed to log files under
    $env:TEMP.

.PARAMETER Config
    Path to a translation-server.config.json. Defaults to the repository's
    translation-server.config.json. Relative paths resolve from the current
    working directory (normal CLI behavior).

.EXAMPLE
    .\Start-TranslationServer.ps1

.EXAMPLE
    .\Start-TranslationServer.ps1 -Config ".\my-config.json"
#>

param(
    [string]$Config
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

Write-Host ""
Write-Host "=== Translation Server ===" -ForegroundColor Cyan

# ---- Resolve paths ----
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot }
             else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$RepoRoot = Split-Path -Parent $ScriptDir
Write-Host "[INFO] Repo root:   $RepoRoot"

# ---- Resolve config ----
if (-not $Config) {
    $ConfigPath = Join-Path $RepoRoot 'translation-server.config.json'
}
else {
    # Relative paths resolve from the current working directory (CLI convention)
    if (-not [System.IO.Path]::IsPathRooted($Config)) {
        $ConfigPath = Join-Path (Get-Location) $Config
    }
    else {
        $ConfigPath = $Config
    }
}

if (-not (Test-Path $ConfigPath)) {
    Write-Error "Server config not found at: $ConfigPath"
    exit 1
}
Write-Host "[INFO] Config:      $ConfigPath"

# ---- Read config for status display ----
$ServerHost = '127.0.0.1'
$ServerPort = 8091
$ServerWorkers = 1
$ServerLogLevel = 'info'
$ModelName = 'facebook/m2m100_418M'
$ModelDevice = 'cuda'
$WarmupOnStart = $true
try {
    $ConfigJson = Get-Content -Raw -Path $ConfigPath | ConvertFrom-Json
    if ($ConfigJson.server.host)        { $ServerHost = [string]$ConfigJson.server.host }
    if ($ConfigJson.server.port)        { $ServerPort = [int]$ConfigJson.server.port }
    if ($ConfigJson.server.workers)     { $ServerWorkers = [int]$ConfigJson.server.workers }
    if ($ConfigJson.server.log_level)   { $ServerLogLevel = [string]$ConfigJson.server.log_level }
    if ($ConfigJson.translation.model_name) { $ModelName = [string]$ConfigJson.translation.model_name }
    if ($ConfigJson.translation.device)     { $ModelDevice = [string]$ConfigJson.translation.device }
    if ($null -ne $ConfigJson.runtime.warmup_on_start) { $WarmupOnStart = [bool]$ConfigJson.runtime.warmup_on_start }
}
catch {
    Write-Host "[WARN] Could not parse config for display (will show defaults): $($_.Exception.Message)" -ForegroundColor Yellow
}

$ServerUrl = "http://${ServerHost}:${ServerPort}"

Write-Host ""
Write-Host "[STATUS] Host:        $ServerHost"
Write-Host "[STATUS] Port:        $ServerPort"
Write-Host "[STATUS] URL:         $ServerUrl"
Write-Host "[STATUS] Workers:     $ServerWorkers"
Write-Host "[STATUS] Log level:   $ServerLogLevel"
Write-Host "[STATUS] Model:       $ModelName"
Write-Host "[STATUS] Device:      $ModelDevice"
Write-Host "[STATUS] Warmup:      $(if ($WarmupOnStart) { 'on start' } else { 'lazy (first request)' })"
Write-Host ""

# ---- Pre-flight: is the port already in use? ----
$listener = Get-NetTCPConnection -LocalPort $ServerPort -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($listener) {
    $ownerName = ''
    $owner = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
    if ($owner) { $ownerName = " ($($owner.ProcessName))" }
    Write-Host ""
    Write-Host "[ERROR] Port $ServerPort is already in use by process $($listener.OwningProcess)$ownerName."
    Write-Host "[ERROR] A translation server may already be running on $ServerUrl."
    Write-Host "[ERROR] Stop it first, or use a different port in the config, then retry."
    exit 3
}

# ---- Locate the 'dp' environment Python ----
# `conda` on PATH may be a real executable or a profile alias/function
# (whose .Source is not a file path). Only accept a real application; the
# known install locations are probed first.
$CondaExe = $null
$cmd = Get-Command conda -ErrorAction SilentlyContinue
if ($cmd -and $cmd.CommandType -eq 'Application' -and $cmd.Source -and (Test-Path $cmd.Source)) {
    $CondaExe = $cmd.Source
}
if (-not $CondaExe) {
    $userPaths = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
        "$env:USERPROFILE\Miniforge3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\Miniconda3\Scripts\conda.exe"
    )
    foreach ($p in $userPaths) {
        if (Test-Path $p) { $CondaExe = $p; break }
    }
}
if (-not $CondaExe -and $env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) {
    $CondaExe = $env:CONDA_EXE
}
if (-not $CondaExe -or -not (Test-Path $CondaExe)) {
    Write-Error "Conda executable not found. Install Miniconda/Anaconda or run .\script\Initialize-Env.ps1 first."
    exit 1
}

# Prefer the environment's python directly (no conda-run output buffering);
# fall back to `conda run --no-capture-output` if it is missing.
$CondaRoot = Split-Path -Parent (Split-Path -Parent $CondaExe)
$EnvPython = Join-Path $CondaRoot "envs\dp\python.exe"
if (Test-Path $EnvPython) {
    $LaunchCmd = $EnvPython
    $LaunchArgs = @('-m', 'translation_server', '-c', $ConfigPath)
}
else {
    Write-Host "[WARN] $EnvPython not found; using 'conda run --no-capture-output'." -ForegroundColor Yellow
    $LaunchCmd = $CondaExe
    $LaunchArgs = @('run', '--no-capture-output', '-n', 'dp', '--cwd', $RepoRoot,
                    'python', '-m', 'translation_server', '-c', $ConfigPath)
}
Write-Host "[INFO] Python:      $LaunchCmd"

# ---- Launch server as a watched child process ----
$OutLog = Join-Path $env:TEMP 'translation-server.out.log'
$ErrLog = Join-Path $env:TEMP 'translation-server.err.log'
$env:PYTHONPATH = Join-Path $RepoRoot 'src'

Write-Host ""
Write-Host "[INFO] Launching translation server on $ServerUrl ..." -ForegroundColor Green
Write-Host "[INFO] First launch may download the model (~1.7 GB). Subsequent starts are fast."
Write-Host "[INFO] Press Ctrl+C to stop the server."
Write-Host ""

$proc = Start-Process -FilePath $LaunchCmd -ArgumentList $LaunchArgs `
    -WorkingDirectory $RepoRoot -PassThru `
    -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog `
    -WindowStyle Hidden

Write-Host "[STATUS] Process:    started (PID $($proc.Id))"
Write-Host "[STATUS] State:      starting ..."
Write-Host ""

# ---- Live status watchdog ----
$startTime = Get-Date
$lastStatusLine = 0
$readySeen = $false
$exitCode = $null

try {
    while (-not $proc.HasExited) {
        $uptime = [int](New-TimeSpan -Start $startTime -End (Get-Date)).TotalSeconds
        $healthOk = $false
        $modelReady = $false
        try {
            $health = Invoke-RestMethod -Uri "$ServerUrl/health" -Method Get -TimeoutSec 3
            $healthOk = $true
            $modelReady = ($health.ready -eq $true)
        }
        catch {
            $healthOk = $false
        }

        $line = "[STATUS] uptime ${uptime}s | port $ServerPort | " +
                $(if ($modelReady) { 'state RUNNING (model ready)' }
                  elseif ($healthOk) { 'state RUNNING (model loading ...)' }
                  else { 'state starting ...' }) +
                $(if ($modelReady) { " | $($health.model) | $($health.device)" } else { '' })

        if ($modelReady -and -not $readySeen) {
            $readySeen = $true
            $secs = [int](New-TimeSpan -Start $startTime -End (Get-Date)).TotalSeconds
            Write-Host "[OK]   Server RUNNING on $ServerUrl (PID $($proc.Id), model $($health.model), device $($health.device), ready in ${secs}s)" -ForegroundColor Green
        }
        elseif ($uptime -ge ($lastStatusLine + 150)) {
            # Periodic status refresh while the server runs (every 2.5 min)
            $lastStatusLine = $uptime
            Write-Host $line
        }
        elseif (-not $readySeen -and $uptime -ge 15 -and $uptime % 15 -eq 0) {
            # Slow start (first model download): keep the user informed
            Write-Host $line -ForegroundColor Yellow
        }

        Start-Sleep -Seconds 2
    }

    $exitCode = $null
    try { $exitCode = $proc.ExitCode } catch { $exitCode = 1 }
    if ($null -eq $exitCode) { $exitCode = 1 }
}
finally {
    if (-not $proc.HasExited) {
        # Give the server a moment to shut down gracefully (e.g. Ctrl+C),
        # then force-terminate the whole process tree if it is still up.
        $proc.WaitForExit(8000) | Out-Null
        if (-not $proc.HasExited) {
            Write-Host "[WARN] Server still running; terminating process tree (PID $($proc.Id))." -ForegroundColor Yellow
            & taskkill /PID $proc.Id /T /F 2>$null | Out-Null
            $proc.WaitForExit(5000) | Out-Null
            try { $exitCode = $proc.ExitCode } catch { $exitCode = 1 }
            if ($null -eq $exitCode) { $exitCode = 1 }
        }
    }
}

# ---- Final status ----
$uptime = [int](New-TimeSpan -Start $startTime -End (Get-Date)).TotalSeconds
Write-Host ""
if ($readySeen -and $exitCode -eq 0) {
    Write-Host "[INFO] Server stopped cleanly (exit code $exitCode, uptime ${uptime}s)." -ForegroundColor Cyan
}
else {
    Write-Host "[ERROR] Server exited with code $exitCode (uptime ${uptime}s)." -ForegroundColor Red
    Write-Host ""
    Write-Host "----- last server log lines (stderr) -----" -ForegroundColor Gray
    if (Test-Path $ErrLog) {
        Get-Content -Tail 15 $ErrLog -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ -ForegroundColor Gray }
    }
    Write-Host "-------------------------------------------" -ForegroundColor Gray
}
exit $exitCode
