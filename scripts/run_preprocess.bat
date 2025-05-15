@echo off
rem 设置UTF-8编码
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

rem 获取参数
set DATASET=%1
set RESOLUTION=%2
set ID_TOKEN=%3
set DECODE_FLAG=%4

rem 输出命令信息
echo 准备执行预处理，参数:
echo 数据集: %DATASET%
echo 分辨率: %RESOLUTION%
echo 触发词: %ID_TOKEN%
echo 解码选项: %DECODE_FLAG%

rem 构建命令
set CMD=C:\ProgramData\anaconda3\python.exe "%~dp0preprocess_dataset.py" "%DATASET%" --resolution-buckets %RESOLUTION% --batch-size 1 --num-workers 2

rem 添加可选参数
if not "%ID_TOKEN%"=="" (
    set CMD=%CMD% --id-token %ID_TOKEN%
)

if "%DECODE_FLAG%"=="--decode-videos" (
    set CMD=%CMD% --decode-videos
)

rem 执行命令
echo 执行命令: %CMD%
%CMD%

rem 返回预处理脚本的错误码
exit /b %errorlevel%
