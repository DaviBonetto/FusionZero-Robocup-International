param(
    [Parameter(Mandatory = $true)]
    [string]$Host,
    [int]$Port = 8765,
    [double]$ReconnectInterval = 1.5,
    [double]$HeartbeatInterval = 1.0,
    [double]$ServerTimeout = 6.0,
    [string]$Python = "",
    [string]$VenvPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-DefaultVenvPath {
    param(
        [string]$ProjectRoot
    )

    if ($env:LOCALAPPDATA) {
        return Join-Path $env:LOCALAPPDATA "FusionZero\venvs\obr_overengineering_v1-pc"
    }

    return Join-Path $ProjectRoot ".venv-pc"
}

function Resolve-PythonInvocation {
    param(
        [string]$Requested,
        [string]$PreferredVenv
    )

    if ($Requested) {
        return @($Requested)
    }

    $VenvPython = Join-Path $PreferredVenv "Scripts\python.exe"
    if (Test-Path $VenvPython) {
        return @($VenvPython)
    }

    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        return @("py", "-3")
    }

    $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCmd) {
        return @($PythonCmd.Source)
    }

    throw "Python 3 not found."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ProjectRoot "..\..")).Path

if (-not $VenvPath) {
    $VenvPath = Resolve-DefaultVenvPath -ProjectRoot $ProjectRoot
}

$PythonInvocation = Resolve-PythonInvocation -Requested $Python -PreferredVenv $VenvPath
$Runner = Join-Path $ProjectRoot "src\remote_dashboard_client.py"

Push-Location $RepoRoot
try {
    & $PythonInvocation[0] @($PythonInvocation | Select-Object -Skip 1) `
        $Runner `
        --host $Host `
        --port $Port `
        --reconnect-interval $ReconnectInterval `
        --heartbeat-interval $HeartbeatInterval `
        --server-timeout $ServerTimeout
}
finally {
    Pop-Location
}
