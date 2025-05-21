#!/bin/bash
# LTX-Video-Trainer Ubuntu Startup Script for AutoDL environments
# Based on Windows script by pipchen
# Date: 2025-05-21

# Set working directory to script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to check if port is available
check_port() {
    nc -z localhost $1 >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        return 1  # Port is in use
    else
        return 0  # Port is available
    fi
}

# Set PYTHONPATH to include the src directory - This fixes the module import issue
echo -e "\e[33mSetting PYTHONPATH to include src directory...\e[0m"
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
echo -e "\e[33mPYTHONPATH = $PYTHONPATH\e[0m"

# Display startup information
echo -e "\e[36mStarting LTX-Video Trainer on AutoDL...\e[0m"
echo -e "\e[33mChecking Python version:\e[0m"
python3 --version

# Try to find an available port starting from 6006
PORT=6006
while ! check_port $PORT && [ $PORT -lt 6016 ]; do
    echo -e "\e[31mPort $PORT is already in use, trying next port...\e[0m"
    PORT=$((PORT+1))
done

if [ $PORT -ge 6016 ]; then
    echo -e "\e[31mCould not find available port in range 6006-6015. Please free up a port and try again.\e[0m"
    exit 1
fi

echo -e "\e[32mStarting interface on port $PORT...\e[0m"

# AutoDL环境提示
echo -e "\e[33m在AutoDL环境中运行注意事项:\e[0m"
echo -e "\e[33m1. 使用screen保持后台运行: screen -S ltx\e[0m"
echo -e "\e[33m2. 在screen中运行: python3 scripts/minimal_ui_run_ubuntu.py --port $PORT --host 0.0.0.0\e[0m"
echo -e "\e[33m3. 分离screen: Ctrl+A 然后按 D\e[0m"
echo -e "\e[33m4. 重新连接: screen -r ltx\e[0m"
echo -e "\e[33m5. 永久后台运行: nohup python3 scripts/minimal_ui_run_ubuntu.py --port $PORT --host 0.0.0.0 > output.log 2>&1 &\e[0m"
echo 

# 询问用户是否使用nohup运行
echo -e "\e[32m选择运行模式:\e[0m"
echo "1) 直接运行 (关闭SSH会话后程序将终止)"
echo "2) 使用nohup后台运行 (关闭SSH会话后程序继续运行)"
read -p "请选择 [1/2]: " choice

case $choice in
    2)
        echo -e "\e[32m使用nohup在后台运行...\e[0m"
        nohup python3 scripts/minimal_ui_run_ubuntu.py --port $PORT --host 0.0.0.0 > output.log 2>&1 &
        echo -e "\e[32m已启动!\e[0m"
        echo -e "\e[32m访问地址: http://127.0.0.1:$PORT\e[0m"
        echo -e "\e[32m日志文件: $SCRIPT_DIR/output.log\e[0m"
        echo -e "\e[32m进程ID: $!\e[0m"
        ;;
    *)
        echo -e "\e[32m直接运行 (按Ctrl+C终止)...\e[0m"
        python3 scripts/minimal_ui_run_ubuntu.py --port $PORT --host 0.0.0.0
        echo -e "\e[36m界面已关闭\e[0m"
        ;;
esac