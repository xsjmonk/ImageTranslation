$ErrorActionPreference = 'Stop'

$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

& "$RepoRoot\script\Initialize-Env.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$env:PYTHONPATH = Join-Path $RepoRoot 'src'
$env:RUN_HYMT2_SMOKE = '1'
conda run -n dp python -m pytest tests/translation/test_hymt2_smoke.py -v -s
exit $LASTEXITCODE
