# LTX-Video-Trainer 系统Python启动脚本
# 作者: pipchen
# 日期: 2025-05-17

# 使用系统Python路径
$pythonPath = "python.exe"

# 设置工作目录为脚本所在位置
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# 显示启动信息
Write-Host "启动LTX-Video训练器..." -ForegroundColor Cyan
Write-Host "使用系统Python: $pythonPath" -ForegroundColor Yellow
Write-Host "检查Python版本:" -ForegroundColor Yellow
& $pythonPath --version

# 启动UI
Write-Host "正在启动界面..." -ForegroundColor Green
& $pythonPath "scripts\minimal_ui.py"

# 程序结束
Write-Host "界面已关闭" -ForegroundColor Cyan
