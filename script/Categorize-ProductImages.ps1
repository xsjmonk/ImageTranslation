param([Parameter(Mandatory=$true)][string]$Config)
$ErrorActionPreference = 'Stop'
$root=Split-Path -Parent $PSScriptRoot; $resolved=(Resolve-Path $Config -ErrorAction Stop).Path
$env:PYTHONPATH=Join-Path $root 'src'
conda run -n dp python -m image_translation.product_image_selection.categorize_cli --config $resolved
exit $LASTEXITCODE
