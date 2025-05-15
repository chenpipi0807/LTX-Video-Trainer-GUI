#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTX-Video-Trainer 简易命令行界面
不依赖任何第三方UI库，使用基本Python实现
"""

import os
import sys
import subprocess
from pathlib import Path

# 默认路径
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(PROJECT_DIR, "configs")
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")

# 读取可用的配置文件
CONFIG_FILES = {}
for file in os.listdir(CONFIGS_DIR):
    if file.endswith(".yaml"):
        name = file.replace(".yaml", "")
        CONFIG_FILES[name] = os.path.join(CONFIGS_DIR, file)

# 预设分辨率
RESOLUTIONS = [
    "512x512x25",
    "768x768x25", 
    "768x768x49",
    "1024x576x41"
]

def run_command(cmd):
    """运行命令并显示输出"""
    print(f"\n执行命令: {' '.join(cmd)}\n")
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line.strip())
        
        process.wait()
        
        if process.returncode == 0:
            print("\n命令执行成功!")
            return True
        else:
            print(f"\n命令执行失败，返回代码 {process.returncode}")
            return False
    
    except Exception as e:
        print(f"\n执行出错: {str(e)}")
        return False

def run_preprocessing():
    """运行数据预处理"""
    print("\n===== 数据预处理 =====")
    
    dataset_path = input("\n数据集路径 (文件夹或元数据文件): ")
    
    print("\n可用分辨率:")
    for i, res in enumerate(RESOLUTIONS, 1):
        print(f"{i}. {res}")
    print(f"{len(RESOLUTIONS) + 1}. 自定义")
    
    res_choice = int(input("\n选择分辨率 (输入编号): "))
    if res_choice <= len(RESOLUTIONS):
        resolution = RESOLUTIONS[res_choice - 1]
    else:
        resolution = input("输入自定义分辨率 (格式: 宽x高x帧数): ")
    
    id_token = input("\nLoRA触发词 (可选，按Enter跳过): ")
    
    decode_videos = input("\n解码视频进行验证? (y/n): ").lower() == "y"
    
    cmd = [
        sys.executable,
        os.path.join(SCRIPTS_DIR, "preprocess_dataset.py"),
        dataset_path,
        "--resolution-buckets", resolution
    ]
    
    if id_token:
        cmd.extend(["--id-token", id_token])
    
    if decode_videos:
        cmd.append("--decode-videos")
    
    run_command(cmd)

def run_training():
    """运行训练脚本"""
    print("\n===== 模型训练 =====")
    
    print("\n可用配置:")
    config_names = list(CONFIG_FILES.keys())
    for i, name in enumerate(config_names, 1):
        print(f"{i}. {name}")
    
    config_choice = int(input("\n选择配置 (输入编号): "))
    config_path = CONFIG_FILES[config_names[config_choice - 1]]
    
    cmd = [
        sys.executable,
        os.path.join(SCRIPTS_DIR, "train.py"),
        config_path
    ]
    
    run_command(cmd)

def run_pipeline():
    """运行完整流水线"""
    print("\n===== 一键训练流水线 =====")
    
    basename = input("\n项目名称 (例如: my_effect): ")
    
    print("\n可用分辨率:")
    for i, res in enumerate(RESOLUTIONS, 1):
        print(f"{i}. {res}")
    print(f"{len(RESOLUTIONS) + 1}. 自定义")
    
    res_choice = int(input("\n选择分辨率 (输入编号): "))
    if res_choice <= len(RESOLUTIONS):
        resolution = RESOLUTIONS[res_choice - 1]
    else:
        resolution = input("输入自定义分辨率 (格式: 宽x高x帧数): ")
    
    print("\n可用配置模板:")
    config_names = list(CONFIG_FILES.keys())
    for i, name in enumerate(config_names, 1):
        print(f"{i}. {name}")
    
    config_choice = int(input("\n选择配置模板 (输入编号): "))
    config_template = CONFIG_FILES[config_names[config_choice - 1]]
    
    rank = int(input("\nLoRA秩 (1-128，推荐32): ") or "32")
    
    cmd = [
        sys.executable,
        os.path.join(SCRIPTS_DIR, "run_pipeline.py"),
        basename,
        "--resolution-buckets", resolution,
        "--config-template", config_template,
        "--rank", str(rank)
    ]
    
    run_command(cmd)

def convert_to_comfyui():
    """转换为ComfyUI格式"""
    print("\n===== 转换为ComfyUI格式 =====")
    
    input_path = input("\n输入模型路径 (.safetensors): ")
    output_path = input("\n输出路径 (可选，按Enter跳过): ")
    
    cmd = [
        sys.executable,
        os.path.join(SCRIPTS_DIR, "convert_checkpoint.py"),
        input_path,
        "--to-comfy"
    ]
    
    if output_path:
        cmd.extend(["--output_path", output_path])
    
    run_command(cmd)

def show_help():
    """显示帮助信息"""
    print("""
===== LTX-Video-Trainer 使用帮助 =====

训练要求:
- 硬件: 推荐使用至少24GB显存的NVIDIA GPU
- 训练数据: 5-50个短视频片段 (每个5-15秒)
- 训练时间: 取决于数据集大小、训练轮数和GPU性能

快速开始:
1. 一键训练流水线
   - 创建名为`项目名称_raw`的文件夹，将视频放入其中
   - 运行一键训练流水线，填写项目名称（不含"_raw"后缀）

2. 自定义工作流
   - 预处理: 准备并处理数据集
   - 训练: 配置并运行训练
   - 转换: 将训练好的权重转换为ComfyUI格式

数据集准备建议:
- 使用5-15秒的短片展示你想要训练的效果
- 确保视频具有一致的质量和风格
- 最好使用多个不同角度/场景的效果示例

配置建议:
- 分辨率: 更高的分辨率需要更多显存但能捕捉更多细节
- 训练轮数: 从100-200轮开始，根据结果调整
- LoRA秩: 更高的秩(16-64)能捕捉更复杂的效果，但需要更多数据

分辨率选择指南:
分辨率格式为"宽x高x帧数"，例如"768x768x49"，其中:
- 宽度和高度必须是32的倍数
- 帧数必须是8的倍数加1（如9、17、25、33等）
""")

def main():
    """主函数"""
    while True:
        print("\n===== LTX-Video训练器 =====")
        print("1. 一键训练流水线")
        print("2. 数据预处理")
        print("3. 模型训练")
        print("4. 转换为ComfyUI格式")
        print("5. 使用帮助")
        print("0. 退出")
        
        choice = input("\n请选择操作 (输入编号): ")
        
        if choice == "1":
            run_pipeline()
        elif choice == "2":
            run_preprocessing()
        elif choice == "3":
            run_training()
        elif choice == "4":
            convert_to_comfyui()
        elif choice == "5":
            show_help()
        elif choice == "0":
            print("\n感谢使用LTX-Video训练器!")
            break
        else:
            print("\n无效的选择，请重试。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作已取消。感谢使用LTX-Video训练器!")
    except Exception as e:
        print(f"\n\n发生错误: {str(e)}")
