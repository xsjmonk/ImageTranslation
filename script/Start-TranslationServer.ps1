$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

<#
.SYNOPSIS
    Bootstrap and start the M2M100 GPU translation server.

.DESCRIPTION
    Ensures the 'dp' Conda environment is ready, then launches the
    translation server in the foreground on http://127.0.0.1:8091.

.PARAMETER Config
    Path to a translation-server.config.json. Defaults to the
    translation-server.config.json in the same directory as this script.

.EXAMPLE
    .\Start-TranslationServer.ps1

.EXAMPLE
    .\Start-TranslationServer.ps1 -Config ".\my-config.json"
#>

param(
    [string]$Config
)

# ---- Resolve paths ----
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot }
             else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$RepoRoot = Split-Path -Parent $ScriptDir

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
    # Absolute paths stay as-is; relative paths resolve from current working dir
    if (-not [System.IO.Path]::IsPathRooted($Config)) {
        $ConfigPath = Join-Path $RepoRoot $Config
    }
    else {
        $ConfigPath = $Config
    }
}

if (-not (Test-Path $ConfigPath)) {
    Write-Error "Server config not found at: $ConfigPath"
    exit 1
}

Write-Host "[INFO] Server config: $ConfigPath"

# ---- Ensure environment ----
Write-Host "[INFO] Verifying Conda environment 'dp'..."
& powershell -ExecutionPolicy Bypass -File $InitScript
if ($LASTEXITCODE -ne 0) {
    Write-Error "Environment initialization failed."
    exit 1
}

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

# ---- Launch server ----
Write-Host "[INFO] Starting translation server on http://127.0.0.1:8091 ..."
Write-Host "[INFO] First launch may download the model (~1.7 GB). Subsequent starts are fast."
Write-Host ""

$env:PYTHONPATH = Join-Path $RepoRoot 'src'

& $CondaExe run -n dp --cwd $RepoRoot python -m translation_server -c $ConfigPath
exit $LASTEXITCODE
