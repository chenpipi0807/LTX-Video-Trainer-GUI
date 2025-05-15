@echo off
REM Simple preprocessor wrapper with minimal text parsing
chcp 65001 >nul

REM Run Python script with parameters passed directly
C:\ProgramData\anaconda3\python.exe "%~dp0preprocess_dataset.py" %*

REM Return exit code
exit /b %errorlevel%
