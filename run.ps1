# LTX-Video-Trainer Launcher Script
# Author: pipchen
# Date: 2025-05-13

# Direct path to Anaconda Python
$pythonPath = "C:\ProgramData\anaconda3\python.exe"

# Set working directory to script location
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Set PYTHONPATH to include the src directory - This fixes the module import issue
Write-Host "Setting PYTHONPATH to include src directory..." -ForegroundColor Yellow
$env:PYTHONPATH = "$scriptPath\src;$env:PYTHONPATH"
Write-Host "PYTHONPATH = $env:PYTHONPATH" -ForegroundColor Yellow

# Display startup info
Write-Host "Starting LTX-Video Trainer..." -ForegroundColor Cyan
Write-Host "Using Python: $pythonPath" -ForegroundColor Yellow

# Launch UI
& $pythonPath "scripts\minimal_ui.py"

# Program ended
Write-Host "UI closed" -ForegroundColor Cyan
