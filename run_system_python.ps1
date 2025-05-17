# LTX-Video-Trainer System Python Startup Script
# Author: pipchen
# Date: 2025-05-17

# Use system Python path
$pythonPath = "python.exe"

# Set working directory to script location
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Display startup information
Write-Host "Starting LTX-Video Trainer..." -ForegroundColor Cyan
Write-Host "Using system Python: $pythonPath" -ForegroundColor Yellow
Write-Host "Checking Python version:" -ForegroundColor Yellow
& $pythonPath --version

# Launch UI
Write-Host "Starting interface..." -ForegroundColor Green
& $pythonPath "scripts\minimal_ui.py"

# Program end
Write-Host "Interface closed" -ForegroundColor Cyan

# Pause to display results, wait for user to press any key to close
Pause
