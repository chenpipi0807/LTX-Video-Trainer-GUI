# Simple FFmpeg local installer
# Downloads FFmpeg and extracts it directly to the project folder

# Function for colored text output
function Write-ColorText {
    param (
        [Parameter(Mandatory=$true)]
        [string]$Text,
        
        [Parameter(Mandatory=$false)]
        [string]$ForegroundColor = "White"
    )
    
    Write-Host $Text -ForegroundColor $ForegroundColor
}

Write-ColorText "Starting FFmpeg local installation..." -ForegroundColor "Cyan"

# Directories
$projectDir = "C:\LTX-Video-Trainer-GUI"
$binDir = Join-Path $projectDir "bin"
$tempDir = Join-Path $env:TEMP "ffmpeg_download"
$zipPath = Join-Path $tempDir "ffmpeg.zip"
$extractPath = Join-Path $tempDir "extracted"

# Create directories
if (-not (Test-Path -Path $tempDir)) {
    Write-ColorText "Creating temp directory..." -ForegroundColor "Yellow"
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
}

if (-not (Test-Path -Path $binDir)) {
    Write-ColorText "Creating bin directory in project folder..." -ForegroundColor "Yellow"
    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
}

if (-not (Test-Path -Path $extractPath)) {
    New-Item -ItemType Directory -Path $extractPath -Force | Out-Null
}

# FFmpeg download URL
$ffmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

# Download FFmpeg
Write-ColorText "Downloading FFmpeg..." -ForegroundColor "Cyan"
try {
    # Use .NET WebClient to download file
    $webClient = New-Object System.Net.WebClient
    $webClient.DownloadFile($ffmpegUrl, $zipPath)
    
    if (Test-Path -Path $zipPath) {
        Write-ColorText "FFmpeg download successful!" -ForegroundColor "Green"
    } else {
        Write-ColorText "FFmpeg download failed" -ForegroundColor "Red"
        exit 1
    }
} catch {
    Write-ColorText "Download error: $_" -ForegroundColor "Red"
    exit 1
}

# Extract FFmpeg
Write-ColorText "Extracting FFmpeg..." -ForegroundColor "Cyan"
try {
    # Use .NET's System.IO.Compression.ZipFile class to extract files
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zipPath, $extractPath)
    
    Write-ColorText "FFmpeg extraction successful!" -ForegroundColor "Green"
} catch {
    Write-ColorText "Extraction error: $_" -ForegroundColor "Red"
    exit 1
}

# Find ffmpeg.exe in the extracted folder
Write-ColorText "Finding FFmpeg executable..." -ForegroundColor "Cyan"
$ffmpegExe = Get-ChildItem -Path $extractPath -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
if ($ffmpegExe) {
    $ffmpegDir = $ffmpegExe.DirectoryName
    Write-ColorText "Found FFmpeg executable at: $ffmpegDir" -ForegroundColor "Green"
    
    # Copy ffmpeg.exe, ffprobe.exe and ffplay.exe to the bin directory
    foreach ($file in @("ffmpeg.exe", "ffprobe.exe", "ffplay.exe")) {
        $sourcePath = Join-Path $ffmpegDir $file
        $destPath = Join-Path $binDir $file
        
        if (Test-Path -Path $sourcePath) {
            Write-ColorText "Copying $file to project bin directory..." -ForegroundColor "Yellow"
            Copy-Item -Path $sourcePath -Destination $destPath -Force
            
            if (Test-Path -Path $destPath) {
                Write-ColorText "$file successfully copied" -ForegroundColor "Green"
            } else {
                Write-ColorText "Failed to copy $file" -ForegroundColor "Red"
            }
        } else {
            Write-ColorText "$file not found in extracted directory" -ForegroundColor "Yellow"
        }
    }
    
    # Create a bat file to update PATH temporarily
    $batPath = Join-Path $projectDir "set_ffmpeg_path.bat"
    $batContent = @"
@echo off
set PATH=%PATH%;$binDir
echo FFmpeg path set to: $binDir
echo You can now run your application in this command window
"@
    
    Set-Content -Path $batPath -Value $batContent
    Write-ColorText "Created set_ffmpeg_path.bat to temporarily set PATH" -ForegroundColor "Green"
    
    # Clean up temp files
    if (Test-Path -Path $zipPath) {
        Remove-Item -Path $zipPath -Force
    }
    
    if (Test-Path -Path $extractPath) {
        Remove-Item -Path $extractPath -Recurse -Force
    }
    
    Write-ColorText "Temporary files cleaned up" -ForegroundColor "Yellow"
    
    # Final instructions
    Write-ColorText "`nFFmpeg installation complete!" -ForegroundColor "Green"
    Write-ColorText "FFmpeg is now available in the project bin directory: $binDir" -ForegroundColor "Cyan"
    Write-ColorText "`nTo use FFmpeg, either:" -ForegroundColor "White"
    Write-ColorText "1. Run set_ffmpeg_path.bat before starting your application, or" -ForegroundColor "White"
    Write-ColorText "2. Modify your Python code to specify the full path to FFmpeg:" -ForegroundColor "White"
    Write-ColorText "   Example: subprocess.run(['$binDir\ffmpeg', ...])" -ForegroundColor "Yellow"
    
    # Create a Python helper module
    $helperPath = Join-Path $projectDir "scripts\ffmpeg_helper.py"
    $helperContent = @"
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FFmpeg Helper Module
Provides helper functions to locate FFmpeg executables in the project directory
"""

import os
import subprocess
import sys

# Project bin directory containing FFmpeg executables
BIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bin')

# Paths to FFmpeg executables
FFMPEG_EXE = os.path.join(BIN_DIR, 'ffmpeg.exe')
FFPROBE_EXE = os.path.join(BIN_DIR, 'ffprobe.exe')
FFPLAY_EXE = os.path.join(BIN_DIR, 'ffplay.exe')

def get_ffmpeg_path():
    """Return the full path to ffmpeg.exe"""
    return FFMPEG_EXE if os.path.exists(FFMPEG_EXE) else 'ffmpeg'

def get_ffprobe_path():
    """Return the full path to ffprobe.exe"""
    return FFPROBE_EXE if os.path.exists(FFPROBE_EXE) else 'ffprobe'

def get_ffplay_path():
    """Return the full path to ffplay.exe"""
    return FFPLAY_EXE if os.path.exists(FFPLAY_EXE) else 'ffplay'

def run_ffmpeg(args):
    """Run FFmpeg with the given arguments"""
    cmd = [get_ffmpeg_path()] + args
    return subprocess.run(cmd, check=True, capture_output=True)

# Test if FFmpeg is available
def test_ffmpeg():
    """Test if FFmpeg is available and return version info"""
    try:
        result = subprocess.run([get_ffmpeg_path(), '-version'], 
                               check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    print(f"FFmpeg path: {get_ffmpeg_path()}")
    print(f"FFmpeg version info:\\n{test_ffmpeg()}")
"@
    
    Set-Content -Path $helperPath -Value $helperContent
    Write-ColorText "Created FFmpeg helper module at: $helperPath" -ForegroundColor "Green"
    
} else {
    Write-ColorText "Could not find ffmpeg.exe in the extracted folder!" -ForegroundColor "Red"
    exit 1
}

Write-ColorText "`nInstallation process completed!" -ForegroundColor "Green"
