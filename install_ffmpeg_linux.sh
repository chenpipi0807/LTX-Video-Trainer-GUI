#!/bin/bash

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}开始安装FFmpeg...${NC}"

# 检测系统
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    OS_ID=$ID
else
    echo -e "${RED}无法检测操作系统类型，尝试使用通用安装方法...${NC}"
    OS="Unknown"
    OS_ID="unknown"
fi

# 检查FFmpeg是否已安装
if command -v ffmpeg >/dev/null 2>&1; then
    echo -e "${GREEN}FFmpeg已安装，版本:${NC}"
    ffmpeg -version | head -n 1
    echo -e "${YELLOW}跳过安装...${NC}"
    exit 0
fi

# 检测是否为root用户
if [ "$(id -u)" -eq 0 ]; then
    # 已经是root用户，不需要sudo
    USE_SUDO=""
else
    # 普通用户，需要使用sudo
    USE_SUDO="sudo"
fi

# 根据系统类型安装FFmpeg
case $OS_ID in
    ubuntu|debian|linuxmint)
        echo -e "${BLUE}检测到 $OS 系统，使用apt安装FFmpeg...${NC}"
        echo -e "${YELLOW}更新软件包列表...${NC}"
        $USE_SUDO apt update
        echo -e "${YELLOW}安装FFmpeg...${NC}"
        $USE_SUDO apt install -y ffmpeg
        ;;
    centos|rhel|fedora)
        echo -e "${BLUE}检测到 $OS 系统，使用yum/dnf安装FFmpeg...${NC}"
        if command -v dnf >/dev/null 2>&1; then
            $USE_SUDO dnf install -y epel-release
            $USE_SUDO dnf install -y ffmpeg
        else
            $USE_SUDO yum install -y epel-release
            $USE_SUDO yum install -y ffmpeg
        fi
        ;;
    arch|manjaro)
        echo -e "${BLUE}检测到 $OS 系统，使用pacman安装FFmpeg...${NC}"
        $USE_SUDO pacman -S --noconfirm ffmpeg
        ;;
    *)
        echo -e "${RED}未知的操作系统类型: $OS${NC}"
        echo -e "${YELLOW}尝试通用安装方法...${NC}"
        # 尝试使用apt（最常见的包管理器）
        if command -v apt >/dev/null 2>&1; then
            $USE_SUDO apt update
            $USE_SUDO apt install -y ffmpeg
        # 尝试使用yum
        elif command -v yum >/dev/null 2>&1; then
            $USE_SUDO yum install -y epel-release
            $USE_SUDO yum install -y ffmpeg
        # 尝试使用dnf
        elif command -v dnf >/dev/null 2>&1; then
            $USE_SUDO dnf install -y epel-release
            $USE_SUDO dnf install -y ffmpeg
        # 尝试使用pacman
        elif command -v pacman >/dev/null 2>&1; then
            $USE_SUDO pacman -S --noconfirm ffmpeg
        else
            echo -e "${RED}无法安装FFmpeg，请手动安装。${NC}"
            exit 1
        fi
        ;;
esac

# 验证安装
if command -v ffmpeg >/dev/null 2>&1; then
    echo -e "${GREEN}FFmpeg安装成功，版本:${NC}"
    ffmpeg -version | head -n 1
else
    echo -e "${RED}FFmpeg安装失败，请检查错误信息并手动安装。${NC}"
    exit 1
fi

echo -e "\n${GREEN}安装完成!${NC}"
