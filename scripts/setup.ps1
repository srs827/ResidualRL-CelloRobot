param(
    [switch]$Hardware,
    [switch]$Labeling,
    [string]$Python
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if ($Python) {
    $PythonCommand = $Python
    $PythonPrefix = @()
} else {
    $PythonCommand = "py"
    $PythonPrefix = @("-3.11")
}

& $PythonCommand @PythonPrefix -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), "Python 3.11 or 3.12 is required"'
if ($LASTEXITCODE -ne 0) { throw "Could not run a supported Python interpreter." }

& $PythonCommand @PythonPrefix -m venv (Join-Path $RepoRoot ".venv")
if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtual environment." }

$Extras = @("dev")
if ($Hardware) { $Extras += "hardware" }
if ($Labeling) { $Extras += "labeling" }
$ExtrasCsv = $Extras -join ","
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to update packaging tools." }

& $VenvPython -m pip install -e "${RepoRoot}[$ExtrasCsv]"
if ($LASTEXITCODE -ne 0) { throw "Failed to install project dependencies." }

& $VenvPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Smoke tests failed." }

Write-Host ""
Write-Host "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1"
