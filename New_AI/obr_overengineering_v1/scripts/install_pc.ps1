param(
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

if (-not $VenvPath) {
    $VenvPath = Resolve-DefaultVenvPath -ProjectRoot $ProjectRoot
}

$Requirements = Join-Path $ProjectRoot "deploy\requirements-pc.txt"
$BootstrapPython = Resolve-PythonInvocation -Requested $Python -PreferredVenv $VenvPath
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvParent = Split-Path -Parent $VenvPath

if ($VenvParent -and -not (Test-Path $VenvParent)) {
    New-Item -ItemType Directory -Path $VenvParent -Force | Out-Null
}

if (-not (Test-Path $VenvPython)) {
    & $BootstrapPython[0] @($BootstrapPython | Select-Object -Skip 1) -m venv $VenvPath
}

& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install -r $Requirements

Write-Host "PC dashboard install complete."
Write-Host "Venv: $VenvPath"
Write-Host "Run with: powershell -ExecutionPolicy Bypass -File `"$ProjectRoot\scripts\run_pc_dashboard.ps1`" -Host <RASPBERRY_IP>"
