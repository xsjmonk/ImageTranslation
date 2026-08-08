$ErrorActionPreference = 'Stop'

# Keep native-command stderr (conda log/progress output) from throwing under
# $ErrorActionPreference='Stop'; command failures are handled via exit codes.
$PSNativeCommandUseErrorActionPreference = $false

# ============================================================
# Initialize-Env.ps1
# Conda environment initialization and validation script
#
# Ensures the 'dp' Conda environment exists with all
# dependencies declared in environment.yml satisfied.
# Safe to run repeatedly.
# ============================================================

function Write-Status {
    param(
        [string]$Level,
        [string]$Message
    )
    $prefix = "[$Level]"
    switch ($Level) {
        'ERROR'            { Write-Host "$prefix $Message" -ForegroundColor Red }
        'MISSING'          { Write-Host "$prefix $Message" -ForegroundColor Yellow }
        'VERSION MISMATCH' { Write-Host "$prefix $Message" -ForegroundColor Magenta }
        'INSTALL'          { Write-Host "$prefix $Message" -ForegroundColor Cyan }
        'OK'               { Write-Host "$prefix $Message" -ForegroundColor Green }
        'INFO'             { Write-Host "$prefix $Message" -ForegroundColor Gray }
        default            { Write-Host "$prefix $Message" }
    }
}

# ----------------------------------------------------------
# Locate a working conda.exe on this machine.
# Tries: Get-Command, CONDA_EXE env var, then common
# per-user and machine-wide install paths.
# ----------------------------------------------------------
function Find-Conda {
    # 1. Already on PATH
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    # 2. CONDA_EXE environment variable
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) {
        return $env:CONDA_EXE
    }

    # 3. Per-user installations
    $userPaths = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
        "$env:USERPROFILE\Miniforge3\Scripts\conda.exe",
        "$env:USERPROFILE\mambaforge\Scripts\conda.exe",
        "$env:USERPROFILE\Mambaforge\Scripts\conda.exe"
    )
    foreach ($p in $userPaths) {
        if (Test-Path $p) { return $p }
    }

    # 4. Machine-wide installations
    $machinePaths = @(
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\Miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\Anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniforge3\Scripts\conda.exe"
    )
    foreach ($p in $machinePaths) {
        if (Test-Path $p) { return $p }
    }

    throw "Conda executable not found. Please install Miniconda or Anaconda and try again."
}

# ----------------------------------------------------------
# Minimal YAML reader that extracts:
#   - the top-level "name:" value
#   - Conda dependencies (top-level list items under "dependencies:")
#   - Pip dependencies (nested list items under "  - pip:")
# Supports only the structure used by this project's environment.yml.
# ----------------------------------------------------------
function Read-EnvironmentYaml {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        throw "environment.yml not found at: $Path"
    }

    $lines = Get-Content $Path
    $condaDeps = @()
    $pipDeps   = @()
    $envName   = $null
    $inDeps    = $false
    $inPip     = $false

    foreach ($line in $lines) {
        $trimmed = $line.Trim()

        # Skip blank lines and comments
        if (-not $trimmed -or $trimmed.StartsWith('#')) { continue }

        # Top-level "name:" key
        if ($trimmed -match '^name:\s*(.+)$') {
            $envName = $Matches[1].Trim()
            continue
        }

        # Enter dependencies block
        if ($trimmed -eq 'dependencies:') {
            $inDeps = $true
            $inPip  = $false
            continue
        }

        # Exit dependencies block when we hit another top-level key
        if ($inDeps -and ($line -match '^[a-zA-Z]')) {
            $inDeps = $false
            $inPip  = $false
            continue
        }

        if (-not $inDeps) { continue }

        # Detect pip: subsection
        if ($trimmed -eq '- pip:') {
            $inPip = $true
            continue
        }

        if ($inPip) {
            # Pip entries are indented further under "- pip:"
            if ($trimmed -match '^-\s+(.+)$') {
                $dep = $Matches[1].Trim()
                if ($dep -and -not $dep.StartsWith('#')) {
                    $pipDeps += $dep
                }
            }
        }
        else {
            # Top-level Conda dependency entries
            if ($trimmed -match '^-\s+(.+)$') {
                $dep = $Matches[1].Trim()
                if ($dep -and -not $dep.StartsWith('#') -and -not $dep.StartsWith('-')) {
                    $condaDeps += $dep
                }
            }
        }
    }

    return @{
        Name      = $envName
        CondaDeps = $condaDeps
        PipDeps   = $pipDeps
    }
}

# ----------------------------------------------------------
# Split a dependency string into a package name and an
# optional version constraint.
#   "python=3.10"        -> Name=python, Constraint="=3.10"
#   "paddlepaddle==3.2.0" -> Name=paddlepaddle, Constraint="==3.2.0"
#   "paddleocr>=3.3,<4"   -> Name=paddleocr, Constraint=">=3.3,<4"
#   "numpy"               -> Name=numpy, Constraint=""
# ----------------------------------------------------------
function Parse-Dependency {
    param([string]$Dep)

    if ($Dep -match '^([a-zA-Z0-9_][a-zA-Z0-9_.-]*?)\s*([=><!].*)$') {
        return @{ Name = $Matches[1]; Constraint = $Matches[2] }
    }
    else {
        return @{ Name = $Dep; Constraint = '' }
    }
}

# ----------------------------------------------------------
# Compare two dot-separated version strings.
# Returns -1, 0, or 1 (like a standard comparator).
# ----------------------------------------------------------
function Compare-Versions {
    param([string]$A, [string]$B)

    $aParts = @()
    foreach ($seg in ($A -split '\.')) {
        $num = 0
        if ([int]::TryParse($seg, [ref]$num)) {
            $aParts += $num
        }
        else {
            $aParts += $seg
        }
    }

    $bParts = @()
    foreach ($seg in ($B -split '\.')) {
        $num = 0
        if ([int]::TryParse($seg, [ref]$num)) {
            $bParts += $num
        }
        else {
            $bParts += $seg
        }
    }

    $maxLen = [Math]::Max($aParts.Count, $bParts.Count)
    for ($i = 0; $i -lt $maxLen; $i++) {
        $av = if ($i -lt $aParts.Count) { $aParts[$i] } else { 0 }
        $bv = if ($i -lt $bParts.Count) { $bParts[$i] } else { 0 }

        if ($av -is [int] -and $bv -is [int]) {
            if ($av -lt $bv) { return -1 }
            if ($av -gt $bv) { return 1 }
        }
        else {
            $cmp = [string]::Compare($av.ToString(), $bv.ToString(), [StringComparison]::OrdinalIgnoreCase)
            if ($cmp -ne 0) { return [Math]::Sign($cmp) }
        }
    }
    return 0
}

# ----------------------------------------------------------
# Test whether an installed version satisfies a constraint.
# Constraint format examples:
#   "=3.10"          Conda fuzzy  (prefix match)
#   "==3.2.0"        Pip exact
#   ">=3.3,<4"       Pip range   (AND-ed)
# ----------------------------------------------------------
function Test-VersionConstraint {
    param([string]$Installed, [string]$Constraint)

    if (-not $Constraint) { return $true }

    $parts = $Constraint -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }

    # NOTE: if/elseif, NOT switch -Regex: patterns overlap (e.g. '^==' and '^='
    # both match '==x', '^>=' and '^>' both match '>=x') and PowerShell switch
    # executes EVERY matching case, which corrupts the comparison result.
    foreach ($part in $parts) {
        if ($part -match '^==(.+)$') {
            if ((Compare-Versions $Installed $Matches[1]) -ne 0) { return $false }
        }
        elseif ($part -match '^>=\s*(.+)$') {
            if ((Compare-Versions $Installed $Matches[1]) -lt 0) { return $false }
        }
        elseif ($part -match '^<=\s*(.+)$') {
            if ((Compare-Versions $Installed $Matches[1]) -gt 0) { return $false }
        }
        elseif ($part -match '^>\s*(.+)$') {
            if ((Compare-Versions $Installed $Matches[1]) -le 0) { return $false }
        }
        elseif ($part -match '^<\s*(.+)$') {
            if ((Compare-Versions $Installed $Matches[1]) -ge 0) { return $false }
        }
        elseif ($part -match '^=\s*(.+)$') {
            # Conda-style fuzzy: installed version must start with constraint segments
            $cv = $Matches[1]
            $installedParts = $Installed -split '\.'
            $constraintParts = $cv -split '\.'
            for ($i = 0; $i -lt $constraintParts.Count; $i++) {
                if ($i -ge $installedParts.Count) { return $false }
                if ($installedParts[$i] -ne $constraintParts[$i]) { return $false }
            }
        }
        else {
            Write-Status 'ERROR' "Unknown constraint format: $part"
            return $false
        }
    }
    return $true
}

# ----------------------------------------------------------
# Check whether a named Conda environment exists.
# ----------------------------------------------------------
function Test-EnvironmentExists {
    param([string]$CondaExe, [string]$EnvName)

    try {
        # Parse only the "envs" array instead of the whole document:
        # 'conda env list --json' can contain keys that differ only by case
        # (e.g. ...\Miniconda3 vs ...\miniconda3), which makes a full
        # ConvertFrom-Json throw "keys with different casing".
        $envListJson = & $CondaExe env list --json | Out-String
        $envsMatch = [regex]::Match($envListJson, '"envs"\s*:\s*(\[.*?\])', 'Singleline')
        if (-not $envsMatch.Success) {
            throw 'Could not locate the "envs" array in conda env list output.'
        }
        $envs = $envsMatch.Groups[1].Value | ConvertFrom-Json
        foreach ($ep in $envs) {
            if ((Split-Path $ep -Leaf) -eq $EnvName) {
                return $true
            }
        }
    }
    catch {
        # If we can't parse the JSON, assume env doesn't exist
    }
    return $false
}

# ----------------------------------------------------------
# Return a hashtable of {package-name: version} for Conda
# packages installed in the named environment.
# ----------------------------------------------------------
function Get-CondaPackageVersions {
    param([string]$CondaExe, [string]$EnvName)

    try {
        # stdout only: conda warnings on stderr must not corrupt the JSON
        $json = & $CondaExe list -n $EnvName --json | Out-String
        $list = $json | ConvertFrom-Json
        $result = @{}
        foreach ($pkg in $list) {
            $result[$pkg.name] = $pkg.version
        }
        return $result
    }
    catch {
        Write-Status 'ERROR' "Failed to query Conda packages: $_"
        return @{}
    }
}

# ----------------------------------------------------------
# Return a hashtable of {package-name: version} for Pip
# packages installed in the named environment.
# ----------------------------------------------------------
function Get-PipPackageVersions {
    param([string]$CondaExe, [string]$EnvName)

    try {
        # stdout only: pip/conda warnings on stderr must not corrupt the JSON
        $json = & $CondaExe run -n $EnvName python -m pip list --format=json | Out-String
        $list = $json | ConvertFrom-Json
        $result = @{}
        foreach ($pkg in $list) {
            $result[$pkg.name] = $pkg.version
        }
        return $result
    }
    catch {
        Write-Status 'ERROR' "Failed to query Pip packages: $_"
        return @{}
    }
}

# ----------------------------------------------------------
# Retrieve the Python version inside the named environment.
# ----------------------------------------------------------
function Get-PythonVersion {
    param([string]$CondaExe, [string]$EnvName)

    $output = & $CondaExe run -n $EnvName python --version 2>&1 | Out-String
    if ($output -match 'Python\s+(\S+)') {
        return $Matches[1]
    }
    throw "Failed to determine Python version inside environment '$EnvName'."
}

# ----------------------------------------------------------
# Look up an installed version of a package by name,
# trying both original name and dash/underscore variants.
# ----------------------------------------------------------
function Find-InstalledVersion {
    param([hashtable]$AllPackages, [string]$Name)

    if ($AllPackages.ContainsKey($Name)) {
        return $AllPackages[$Name]
    }
    # Try underscore <-> dash swap
    $alt = $Name -replace '-', '_'
    if ($alt -ne $Name -and $AllPackages.ContainsKey($alt)) {
        return $AllPackages[$alt]
    }
    $alt = $Name -replace '_', '-'
    if ($alt -ne $Name -and $AllPackages.ContainsKey($alt)) {
        return $AllPackages[$alt]
    }
    return $null
}

# ----------------------------------------------------------
# Check all declared dependencies and return a list of
# issue objects (MISSING / VERSION_MISMATCH).
# Also writes [OK] / [MISSING] / [VERSION MISMATCH] lines.
# ----------------------------------------------------------
function Confirm-Dependencies {
    param(
        [array]  $AllDeps,
        [hashtable]$AllPackages
    )

    $issues = @()

    foreach ($dep in $AllDeps) {
        $name       = $dep.Name
        $constraint = $dep.Constraint

        $installedVersion = Find-InstalledVersion -AllPackages $AllPackages -Name $name

        if (-not $installedVersion) {
            Write-Status 'MISSING' $name
            $issues += @{ Name = $name; Issue = 'MISSING' }
        }
        elseif (-not (Test-VersionConstraint -Installed $installedVersion -Constraint $constraint)) {
            Write-Status 'VERSION MISMATCH' "$name installed=$installedVersion required=$constraint"
            $issues += @{ Name = $name; Issue = 'VERSION_MISMATCH'; Installed = $installedVersion; Required = $constraint }
        }
        else {
            $constraintDisplay = if ($constraint) { " satisfies $constraint" } else { '' }
            Write-Status 'OK' "$name $installedVersion$constraintDisplay"
        }
    }

    return $issues
}

# ============================================================
# Main
# ============================================================
function Main {
    Write-Status 'INFO' '=== Conda Environment Initialization ==='

    # ---- 1. Resolve script directory and repo root ----
    $ScriptDir = if ($PSScriptRoot) { $PSScriptRoot }
                 else { Split-Path -Parent $MyInvocation.MyCommand.Path }
    $RepoRoot = Split-Path -Parent $ScriptDir
    Write-Status 'INFO' "Repo root: $RepoRoot"

    # ---- 2. Locate environment.yml ----
    $envYamlPath = Join-Path $RepoRoot 'environment.yml'
    if (-not (Test-Path $envYamlPath)) {
        Write-Status 'ERROR' "environment.yml not found at: $envYamlPath"
        exit 1
    }
    Write-Status 'OK' 'environment.yml found.'

    # ---- 3. Read environment.yml ----
    Write-Status 'INFO' 'Reading environment.yml...'
    $yaml    = Read-EnvironmentYaml -Path $envYamlPath
    $envName = $yaml.Name
    if (-not $envName) {
        Write-Status 'ERROR' "No environment name found in environment.yml"
        exit 1
    }
    Write-Status 'INFO' "Environment name: $envName"
    Write-Status 'INFO' "Conda dependencies declared: $($yaml.CondaDeps.Count)"
    Write-Status 'INFO' "Pip dependencies declared: $($yaml.PipDeps.Count)"

    # ---- 4. Locate Conda ----
    $condaExe = Find-Conda
    Write-Status 'INFO' "Using Conda: $condaExe"

    $condaVersion = & $condaExe --version 2>&1 | Out-String
    Write-Status 'INFO' "Conda version: $($condaVersion.Trim())"

    # ---- 5. Ensure environment exists ----
    $envExists = Test-EnvironmentExists -CondaExe $condaExe -EnvName $envName

    if (-not $envExists) {
        Write-Status 'INFO' "Environment '$envName' does not exist. Creating..."
        Write-Status 'INSTALL' "Running: conda env create -n $envName -f $envYamlPath"

        # Stream conda's own progress/status output live to the console
        & $condaExe env create -n $envName -f $envYamlPath
        if ($LASTEXITCODE -ne 0) {
            Write-Status 'ERROR' "Environment creation failed (exit code $LASTEXITCODE)."
            exit 1
        }
        Write-Status 'OK' "Environment '$envName' created successfully."

        # Double-check it is now visible
        if (-not (Test-EnvironmentExists -CondaExe $condaExe -EnvName $envName)) {
            Write-Status 'ERROR' "Environment creation reported success but environment '$envName' is still not found."
            exit 1
        }
    }
    else {
        Write-Status 'OK' "Conda environment '$envName' already exists."
    }

    # ---- 6. Python version check ----
    $pythonVersion = Get-PythonVersion -CondaExe $condaExe -EnvName $envName
    Write-Status 'OK' "python $pythonVersion"

    # ---- 7. Collect installed packages ----
    $condaPkgs = Get-CondaPackageVersions -CondaExe $condaExe -EnvName $envName
    $pipPkgs   = Get-PipPackageVersions   -CondaExe $condaExe -EnvName $envName

    $allPkgs = @{}
    foreach ($k in $condaPkgs.Keys) { $allPkgs[$k] = $condaPkgs[$k] }
    foreach ($k in $pipPkgs.Keys) {
        if (-not $allPkgs.ContainsKey($k)) {
            $allPkgs[$k] = $pipPkgs[$k]
        }
    }

    # ---- 8. Build dependency list ----
    $allDeps = @()
    foreach ($dep in $yaml.CondaDeps) {
        $parsed = Parse-Dependency -Dep $dep
        $allDeps += @{ Name = $parsed.Name; Constraint = $parsed.Constraint; Source = 'conda' }
    }
    foreach ($dep in $yaml.PipDeps) {
        $parsed = Parse-Dependency -Dep $dep
        $allDeps += @{ Name = $parsed.Name; Constraint = $parsed.Constraint; Source = 'pip' }
    }

    # ---- 9. First validation pass ----
    Write-Status 'INFO' 'Validating dependencies...'
    $issues = Confirm-Dependencies -AllDeps $allDeps -AllPackages $allPkgs

    # ---- 10. Reconcile if needed ----
    if ($issues.Count -gt 0) {
        Write-Status 'INSTALL' "Updating environment '$envName' from environment.yml..."

        # Stream conda's own progress/status output live to the console
        & $condaExe env update -n $envName -f $envYamlPath
        if ($LASTEXITCODE -ne 0) {
            Write-Status 'ERROR' "Environment update failed (exit code $LASTEXITCODE)."
            exit 1
        }
        Write-Status 'OK' 'Environment update completed.'

        # Re-collect packages and re-validate
        Write-Status 'INFO' 'Re-validating dependencies after update...'
        $condaPkgs = Get-CondaPackageVersions -CondaExe $condaExe -EnvName $envName
        $pipPkgs   = Get-PipPackageVersions   -CondaExe $condaExe -EnvName $envName

        $allPkgs = @{}
        foreach ($k in $condaPkgs.Keys) { $allPkgs[$k] = $condaPkgs[$k] }
        foreach ($k in $pipPkgs.Keys) {
            if (-not $allPkgs.ContainsKey($k)) { $allPkgs[$k] = $pipPkgs[$k] }
        }

        $issues = Confirm-Dependencies -AllDeps $allDeps -AllPackages $allPkgs

        if ($issues.Count -gt 0) {
            Write-Status 'ERROR' "$($issues.Count) dependency issue(s) remain unresolved after update."
            exit 1
        }
    }

    Write-Status 'OK' 'All environment.yml dependencies are satisfied.'
    Write-Status 'INFO' '=== Environment initialization complete ==='

    Write-Host ""
    Write-Host "To run commands inside this environment:" -ForegroundColor Gray
    Write-Host "  conda run -n $envName python <script>" -ForegroundColor Gray
    Write-Host ""
}

# ---- Entry point ----
try {
    Main
    exit 0
}
catch {
    Write-Status 'ERROR' $_.Exception.Message
    exit 1
}
