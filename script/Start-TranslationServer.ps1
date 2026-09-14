<#
.SYNOPSIS
    Start the NLLB GPU translation server with a live status panel.

.DESCRIPTION
    Launches the translation server using the existing 'dp' Conda
    environment and shows live status: port, process state (starting /
    running / stopped), model readiness, device, and uptime. Assumes the
    environment is already initialized (run .\script\Initialize-Env.ps1
    separately when needed). Server output is streamed to log files under
    $env:TEMP.

    The server binds to the host configured in translation-server.config.json
    (default 0.0.0.0 = all interfaces, so remote machines on the LAN can
    reach it). When bound to all interfaces the script lists the machine's
    remote URLs and prints a Windows Firewall hint for inbound access.

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

# ---- Locate the 'dp' environment Python (needed before config inspection) ----
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

$CondaExePath = [System.IO.Path]::GetFullPath($CondaExe)
$CondaParent = Split-Path -Parent $CondaExePath
if ((Split-Path -Leaf $CondaParent) -eq 'bin' -and
    (Split-Path -Leaf (Split-Path -Parent $CondaParent)) -eq 'Library') {
    $CondaRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $CondaExePath))
}
else {
    $CondaRoot = Split-Path -Parent $CondaParent
}
$EnvPython = Join-Path $CondaRoot "envs\dp\python.exe"
if (Test-Path $EnvPython) {
    $LaunchCmd = $EnvPython
    $LaunchArgs = @('-m', 'translation_server', '-c', $ConfigPath)
    $SummaryCmd = $EnvPython
    $SummaryArgs = @('-m', 'translation_server', '--print-config-summary', '-c', $ConfigPath)
}
else {
    Write-Host "[WARN] $EnvPython not found; using 'conda run --no-capture-output'." -ForegroundColor Yellow
    $LaunchCmd = $CondaExe
    $LaunchArgs = @('run', '--no-capture-output', '-n', 'dp', '--cwd', $RepoRoot,
                    'python', '-m', 'translation_server', '-c', $ConfigPath)
    $SummaryCmd = $CondaExe
    $SummaryArgs = @('run', '--no-capture-output', '-n', 'dp', '--cwd', $RepoRoot,
                     'python', '-m', 'translation_server', '--print-config-summary', '-c', $ConfigPath)
}
Write-Host "[INFO] Python:      $LaunchCmd"

# ---- Load normalized config summary via the same Python loader as the server ----
$env:PYTHONPATH = Join-Path $RepoRoot 'src'
$summaryJson = & $SummaryCmd @SummaryArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Could not load config summary from $ConfigPath`: $summaryJson"
    exit 1
}
try {
    $ConfigSummary = $summaryJson | ConvertFrom-Json
}
catch {
    Write-Error "Config summary was not valid JSON: $summaryJson"
    exit 1
}

$ServerHost = [string]$ConfigSummary.host
$ServerPort = [int]$ConfigSummary.port
$ServerWorkers = [int]$ConfigSummary.workers
$ServerLogLevel = [string]$ConfigSummary.log_level
$ActiveModel = [string]$ConfigSummary.active_model
$ModelBackend = [string]$ConfigSummary.backend
$ModelName = [string]$ConfigSummary.model
$ModelFamily = [string]$ConfigSummary.model_family
$SourceLanguage = [string]$ConfigSummary.source_language
$TargetLanguage = [string]$ConfigSummary.target_language
$ModelCacheDir = [string]$ConfigSummary.cache_dir
$ModelDevice = [string]$ConfigSummary.device
$Precision = [string]$ConfigSummary.precision
$WarmupOnStart = [bool]$ConfigSummary.warmup_on_start

$ServerUrl = "http://${ServerHost}:${ServerPort}"

# When bound to all interfaces (0.0.0.0 / ::), list the machine's LAN
# addresses so remote clients know which URL to use.
$LanUrls = @()
if ($ServerHost -in @('0.0.0.0', '::')) {
    $LanUrls = @(
        Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object {
                $_.IPAddress -notlike '127.*' -and
                $_.IPAddress -notlike '169.254.*' -and
                $_.PrefixOrigin -ne 'WellKnown'
            } |
            ForEach-Object { "http://$($_.IPAddress):$ServerPort" }
    ) | Sort-Object -Unique
}

Write-Host ""
Write-Host "[STATUS] Host:        $ServerHost"
Write-Host "[STATUS] Port:        $ServerPort"
Write-Host "[STATUS] Local URL:   $ServerUrl"
if ($LanUrls.Count -gt 0) {
    foreach ($u in $LanUrls) {
        Write-Host "[STATUS] Remote URL:  $u"
    }
    Write-Host "[INFO]  Remote access: allow inbound TCP port $ServerPort in Windows Firewall"
    Write-Host "       (run as Administrator: New-NetFirewallRule -DisplayName 'TranslationServer' -Direction Inbound -Protocol TCP -LocalPort $ServerPort -Action Allow)"
}
Write-Host "[STATUS] Workers:     $ServerWorkers"
Write-Host "[STATUS] Log level:   $ServerLogLevel"
Write-Host "[STATUS] Profile:     $ActiveModel"
Write-Host "[STATUS] Backend:     $ModelBackend"
Write-Host "[STATUS] Model:       $ModelName"
Write-Host "[STATUS] Family:      $ModelFamily"
Write-Host "[STATUS] Languages:   $SourceLanguage -> $TargetLanguage"
Write-Host "[STATUS] Cache:       $ModelCacheDir"
Write-Host "[STATUS] Device:      $ModelDevice"
Write-Host "[STATUS] Precision:   $Precision"
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

# ---- Launch server as a watched child process ----
$OutLog = Join-Path $env:TEMP 'translation-server.out.log'
$ErrLog = Join-Path $env:TEMP 'translation-server.err.log'

Write-Host ""
if ($LanUrls.Count -gt 0) {
    Write-Host "[INFO] Launching translation server on all interfaces (local: $ServerUrl) ..." -ForegroundColor Green
}
else {
    Write-Host "[INFO] Launching translation server on $ServerUrl ..." -ForegroundColor Green
}
Write-Host "[INFO] First launch may download the configured model. Subsequent starts reuse the cache."
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
$lastStatusLine = -240      # forces the first periodic line at $FirstStatusAt
$FirstStatusAt = 30         # first state line (incl. download indicator) ~30s
$StatusInterval = 240       # periodic status refresh every 4 min (3-5 min range)
$readySeen = $false
$exitCode = $null

function Get-ServerStateText {
    param([bool]$HealthOk, [bool]$ModelReady)
    if ($ModelReady) { return 'state RUNNING (model ready)' }
    if ($HealthOk) { return 'state RUNNING (model loading ...)' }
    # Tail the server log to detect an in-progress model download
    # (the translator and Hugging Face logs report download progress;
    # 'Model download COMPLETE' marks the end of the download phase).
    $log = ''
    if (Test-Path $ErrLog) {
        $log = (Get-Content -Tail 40 $ErrLog -ErrorAction SilentlyContinue) -join "`n"
    }
    if ($log -match 'Model download COMPLETE') { return 'state model download complete; loading model ...' }
    if ($log -match 'downloading|Fetching \d+ files|Downloading') {
        return 'state DOWNLOADING configured model ...'
    }
    return 'state starting ...'
}

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

        $stateText = Get-ServerStateText -HealthOk $healthOk -ModelReady $modelReady
        $line = "[STATUS] uptime ${uptime}s | port $ServerPort | $stateText" +
                $(if ($modelReady) {
                    " | $($health.model_family) $($health.model) | " +
                    "$($health.source_language)->$($health.target_language) | " +
                    "$($health.device) $($health.precision) | cache $($health.cache_dir)"
                } else { '' })

        if ($modelReady -and -not $readySeen) {
            $readySeen = $true
            $secs = [int](New-TimeSpan -Start $startTime -End (Get-Date)).TotalSeconds
            Write-Host "[OK]   Server RUNNING on $ServerUrl (PID $($proc.Id), model $($health.model), family $($health.model_family), device $($health.device), precision $($health.precision), ready in ${secs}s)" -ForegroundColor Green
        }
        elseif ($uptime -ge ($lastStatusLine + $StatusInterval)) {
            # Periodic status refresh (every 4 min)
            $lastStatusLine = $uptime
            Write-Host $line
        }
        elseif (-not $readySeen -and $uptime -eq $FirstStatusAt) {
            # First state announcement (e.g. model download started)
            $lastStatusLine = $uptime
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
