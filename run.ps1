# LTX-Video-Trainer Launcher Script
# Author: pipchen
# Date: 2025-05-13

# Direct path to Anaconda Python
$pythonPath = "C:\ProgramData\anaconda3\python.exe"

# Set working directory to script location
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Display startup info
Write-Host "Starting LTX-Video Trainer..." -ForegroundColor Cyan
Write-Host "Using Python: $pythonPath" -ForegroundColor Yellow

# Launch UI
& $pythonPath "scripts\minimal_ui.py"

# Program ended
Write-Host "UI closed" -ForegroundColor Cyan
