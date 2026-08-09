<#
.SYNOPSIS
    Bootstrap and start the M2M100 GPU translation server.

.DESCRIPTION
    Ensures the 'dp' Conda environment is ready, verifies CUDA, then launches
    the translation server in the foreground. Shows the configured address
    (host/port) before startup and the exit status on shutdown.

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

$InitScript = Join-Path $RepoRoot 'script\Initialize-Env.ps1'
if (-not (Test-Path $InitScript)) {
    Write-Error "Initialize-Env.ps1 not found at: $InitScript"
    exit 1
}

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

Write-Host ""
Write-Host "[STATUS] Host:        $ServerHost"
Write-Host "[STATUS] Port:        $ServerPort"
Write-Host "[STATUS] URL:         http://$ServerHost`:$ServerPort"
Write-Host "[STATUS] Workers:     $ServerWorkers"
Write-Host "[STATUS] Log level:   $ServerLogLevel"
Write-Host "[STATUS] Model:       $ModelName"
Write-Host "[STATUS] Device:      $ModelDevice"
Write-Host "[STATUS] Warmup:      $(if ($WarmupOnStart) { 'on start' } else { 'lazy (first request)' })"
Write-Host ""

# ---- Ensure environment (run directly in the current PowerShell host) ----
Write-Host "[INFO] Verifying Conda environment 'dp'..."
& $InitScript
if ($LASTEXITCODE -ne 0) {
    Write-Error "Environment initialization failed."
    exit 1
}
Write-Host "[OK]   Conda environment 'dp' is ready."

# ---- Locate Conda ----
$CondaExe = $null
$cmd = Get-Command conda -ErrorAction SilentlyContinue
if ($cmd) { $CondaExe = $cmd.Source }
if (-not $CondaExe -and $env:CONDA_EXE) { $CondaExe = $env:CONDA_EXE }
if (-not $CondaExe) {
    $userPaths = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Miniconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe"
    )
    foreach ($p in $userPaths) {
        if (Test-Path $p) { $CondaExe = $p; break }
    }
}
if (-not $CondaExe) {
    Write-Error "Conda executable not found."
    exit 1
}
Write-Host "[INFO] Conda:        $CondaExe"

# ---- Check GPU (quick CUDA probe) ----
Write-Host "[INFO] Probing CUDA..."
$gpuName = & $CondaExe run -n dp python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')" 2>$null
$gpuName = ($gpuName | Select-Object -Last 1).ToString().Trim()
if ($gpuName -and $gpuName -ne 'NO CUDA') {
    Write-Host "[OK]   GPU:          $gpuName"
}
else {
    Write-Host "[WARN] CUDA not detected. Translation requires an NVIDIA GPU by default." -ForegroundColor Yellow
}

# ---- Launch server (Python logs the configured host/port) ----
Write-Host ""
Write-Host "[INFO] Launching translation server on http://$ServerHost`:$ServerPort ..." -ForegroundColor Green
Write-Host "[INFO] First launch may download the model (~1.7 GB). Subsequent starts are fast."
Write-Host "[INFO] Press Ctrl+C to stop the server."
Write-Host ""

$env:PYTHONPATH = Join-Path $RepoRoot 'src'

& $CondaExe run -n dp --cwd $RepoRoot python -m translation_server -c $ConfigPath
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "[INFO] Translation server stopped (exit code $exitCode)." -ForegroundColor Cyan
}
else {
    Write-Host "[ERROR] Translation server exited with code $exitCode." -ForegroundColor Red
}
exit $exitCode
