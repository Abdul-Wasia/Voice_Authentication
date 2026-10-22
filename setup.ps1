$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Checking Python runtime..."
$pythonCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
}
else {
    throw "Python is not installed or not available on PATH."
}

$versionOutput = & $pythonCmd -c "import sys; print(sys.version)"
Write-Host $versionOutput

if ($versionOutput -match "3\.11") {
    Write-Host "Python 3.11 detected. Using it for the project virtual environment."
}
else {
    Write-Host "Python 3.11 was not found. Trying the system default interpreter for compatibility checks."
}

$venvPath = Join-Path $projectRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..."
    & $pythonCmd -3.11 -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the virtual environment."
    }
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvActivate = Join-Path $venvPath "Scripts\Activate.ps1"

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment Python not found at $venvPython"
}

Write-Host "Activating project virtual environment..."
. $venvActivate

Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip in the project environment."
}

Write-Host "Installing project requirements..."
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install one or more project dependencies."
}

Write-Host "Verifying imports..."
& $venvPython -c "import streamlit, numpy, scipy, sounddevice, soundfile, librosa, noisereduce, sklearn, pandas, matplotlib, bcrypt, torch, torchaudio, speechbrain, resemblyzer; print('ALL CORE IMPORTS OK')"
if ($LASTEXITCODE -ne 0) {
    throw "One or more required imports failed in the project environment."
}

Write-Host "Running lightweight test suite..."
& $venvPython -m pytest tests -q
if ($LASTEXITCODE -ne 0) {
    throw "The project test suite failed."
}

Write-Host "Environment ready. Run the app with:"
Write-Host "  .\.venv\Scripts\python -m streamlit run app.py"
