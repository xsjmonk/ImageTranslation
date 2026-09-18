<#
.SYNOPSIS
    Validate the dp conda environment for local image-processing dependencies.

.DESCRIPTION
    Read-only check. Resolves the ImageTranslation repository and dp Python
    executable explicitly; does not depend on the caller's current directory.

    Requires an initialized conda environment:
      .\script\Initialize-Env.ps1

.EXAMPLE
    .\Check-LocalImageEnv.ps1
#>

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$RepoRoot = Split-Path -Parent $ScriptDir
$SrcRoot = Join-Path $RepoRoot 'src'
$EnvName = if ($env:IMAGE_TRANSLATION_CONDA_ENV) { $env:IMAGE_TRANSLATION_CONDA_ENV } else { 'dp' }

function Write-LauncherError {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

if (-not (Test-Path (Join-Path $RepoRoot 'environment.yml'))) {
    Write-LauncherError "ImageTranslation repository not found at: $RepoRoot"
    Write-LauncherError "Set IMAGE_TRANSLATION_REPO to the repository root and retry."
    exit 1
}

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
    Write-LauncherError "Conda executable not found."
    Write-LauncherError "Install Miniconda/Anaconda, then run: $RepoRoot\script\Initialize-Env.ps1"
    exit 1
}

$CondaExePath = [System.IO.Path]::GetFullPath($CondaExe)
$CondaParent = Split-Path -Parent $CondaExePath
if ((Split-Path -Leaf $CondaParent) -eq 'bin' -and
    (Split-Path -Leaf (Split-Path -Parent $CondaParent)) -eq 'Library') {
    $CondaRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $CondaExePath))
}
else {
    $CondaRoot = Split-Path -Parent $CondaParent
}
$EnvPython = Join-Path $CondaRoot "envs\$EnvName\python.exe"
if (-not (Test-Path $EnvPython)) {
    Write-LauncherError "Conda environment '$EnvName' not found at: $EnvPython"
    Write-LauncherError "Run: $RepoRoot\script\Initialize-Env.ps1"
    exit 1
}

Write-Host "[INFO] Repo root: $RepoRoot" -ForegroundColor Gray
Write-Host "[INFO] Python:    $EnvPython" -ForegroundColor Gray

$env:PYTHONPATH = $SrcRoot
$env:IMAGE_TRANSLATION_REPO = $RepoRoot

& $EnvPython -m image_translation.local_env --check-env
exit $LASTEXITCODE
