@echo off
echo Starting FFmpeg download and installation...
cd /d %~dp0scripts
powershell.exe -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process; .\install_ffmpeg_local.ps1"
echo.
echo Installation completed! Press any key to exit...
pause
