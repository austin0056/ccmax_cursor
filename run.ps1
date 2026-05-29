# Start the OpenAI<->Anthropic proxy.
# Usage:  ./run.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtualenv (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -r requirements.txt
Write-Host "Dependencies ready. Starting proxy..." -ForegroundColor Green
& ".\.venv\Scripts\python.exe" -m proxy.main
