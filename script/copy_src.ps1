$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

<#
.SYNOPSIS
    Copy the ImageTranslation repo to the current folder, zip it, and clean up.

.DESCRIPTION
    - Works from any working directory (repo root resolved from $PSScriptRoot).
    - Copies the whole repo (excluding .reasonix, __pycache__, .git, caches)
      into <current-dir>\ImageTranslation.
    - Deletes all *.pyc files from the copy.
    - Creates <current-dir>\ImageTranslation.zip.
    - Removes the copied folder afterwards.

.EXAMPLE
    cd D:\backup
    & D:\Drop\outlook.com\LocalBox\ImageTranslation\script\copy_src.ps1
#>

# ---- Resolve repo root from script location ----
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot }
             else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$RepoRoot = Split-Path -Parent $ScriptDir
$RepoName = Split-Path -Leaf $RepoRoot
$DestDir  = Join-Path (Get-Location) $RepoName
$ZipPath  = Join-Path (Get-Location) "$RepoName.zip"

Write-Host "[INFO] Repo root:   $RepoRoot"
Write-Host "[INFO] Working dir: $(Get-Location)"
Write-Host "[INFO] Copy to:     $DestDir"

# ---- Clean previous copy/zip (idempotent re-runs) ----
if (Test-Path $DestDir) {
    Write-Host "[WARN] Removing existing destination: $DestDir"
    Remove-Item -Path $DestDir -Recurse -Force
}
if (Test-Path $ZipPath) {
    Write-Host "[WARN] Removing existing zip: $ZipPath"
    Remove-Item -Path $ZipPath -Force
}

# ---- Copy repo (keep structure, exclude noise) ----
$ExcludeDirs = @(
    '.git', '.reasonix', '__pycache__', '.pytest_cache',
    '.idea', '.vscode', 'node_modules', '.mypy_cache'
)

# Create destination FIRST so Copy-Item copies each item INTO it as a subfolder
# (if the destination doesn't exist, Copy-Item copies a directory's *contents*
# instead of the directory itself, which would flatten script\ to the top level)
New-Item -ItemType Directory -Path $DestDir -Force | Out-Null

Write-Host "[INFO] Copying repo..."
foreach ($item in Get-ChildItem -Path $RepoRoot -Force) {
    if ($ExcludeDirs -contains $item.Name) {
        Write-Host "[SKIP] $($item.Name)"
        continue
    }
    # Never copy the destination into itself (happens when run from repo root)
    if ($item.FullName -eq $DestDir) {
        Write-Host "[SKIP] $($item.Name) (destination)"
        continue
    }
    Copy-Item -Path $item.FullName -Destination $DestDir -Recurse -Force
}

# ---- Delete all *.pyc and __pycache__ inside the copy ----
Write-Host "[INFO] Removing *.pyc files..."
Get-ChildItem -Path $DestDir -Recurse -Filter '*.pyc' -Force |
    Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $DestDir -Recurse -Directory -Filter '__pycache__' -Force |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# ---- Zip the copy ----
Write-Host "[INFO] Zipping to: $ZipPath"
Compress-Archive -Path $DestDir -DestinationPath $ZipPath -CompressionLevel Optimal

# ---- Remove the copied folder ----
Write-Host "[INFO] Removing copied folder..."
Remove-Item -Path $DestDir -Recurse -Force

Write-Host "[OK] Done: $ZipPath"
