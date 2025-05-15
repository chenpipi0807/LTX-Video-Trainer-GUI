#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版UI界面，专注于离线训练功能
提供详细的训练日志输出和进度反馈
支持配置文件、数据预处理和ComfyUI格式转换
"""

import os
import sys
import logging
import json
import yaml
import threading
import subprocess
import numpy as np
import torch
import gradio as gr
from pathlib import Path
from safetensors.numpy import save_file

# 设置环境变量强制离线模式
os.environ["PYTHONIOENCODING"] = "utf8"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 获取项目根目录和其他重要目录
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")
CONFIG_DIR = os.path.join(PROJECT_DIR, "config")
DATA_DIR = os.path.join(PROJECT_DIR, "train_date")

sys.path.append(PROJECT_DIR)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('LTX-Trainer')

# 分辨率选项
RESOLUTIONS = [
    "512x512x25", "576x576x25", "640x640x25", "704x704x25", "768x768x25",
    "512x512x49", "576x576x49", "640x640x49", "704x704x49", "768x768x49",
    "576x1024x41", "1024x576x41"
]

# 可能的LTX模型路径
POSSIBLE_MODEL_PATHS = [
    # 项目内部路径
    os.path.join(PROJECT_DIR, "models"),
    # ComfyUI路径
    r"C:\NEWCOMFYUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints",
]

# 读取配置文件
def load_config(config_path):
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"读取配置文件失败: {str(e)}")
        return None

# 保存配置文件
def save_config(config, config_path):
    """保存配置到YAML文件"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        logger.error(f"保存配置文件失败: {str(e)}")
        return False

# 获取所有配置文件
def get_config_files():
    """读取可用的配置文件"""
    config_files = {}
    try:
        if os.path.exists(CONFIG_DIR):
            for file in os.listdir(CONFIG_DIR):
                if file.endswith(".yaml"):
                    name = file.replace(".yaml", "")
                    config_files[name] = os.path.join(CONFIG_DIR, file)
        return config_files
    except Exception as e:
        logger.error(f"读取配置文件列表失败: {str(e)}")
        return {}

def check_dataset_location(basename):
    """检查数据集位置并返回有效的路径"""
    # 首先检查train_date目录
    train_date_path = os.path.join(PROJECT_DIR, "train_date", basename)
    if os.path.exists(train_date_path) and os.listdir(train_date_path):
        logger.info(f"找到train_date目录下的数据集: {train_date_path}")
        return train_date_path
        
    # 然后检查项目根目录下的_raw目录
    raw_path = os.path.join(PROJECT_DIR, f"{basename}_raw")
    if os.path.exists(raw_path) and os.listdir(raw_path):
        logger.info(f"找到原始结构数据集: {raw_path}")
        return raw_path
        
    # 如果都不存在，返回None
    return None

def run_command(command, status=None, verbose=True):
    """运行命令并实时更新状态"""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    output = []
    for line in process.stdout:
        if verbose:
            print(line, end='')
        output.append(line.strip())
        if status and hasattr(status, 'update') and len(output) % 5 == 0:
            status.update(value="\n".join(output[-25:]))
    
    process.wait()
    return process.returncode, "\n".join(output)

def run_offline_training(basename, model_size, resolution, rank, steps, status):
    """
    运行完全离线的训练流程，实时显示日志
    
    Args:
        basename: 项目名称
        model_size: 模型大小 (2B 或 13B)
        resolution: 分辨率
        rank: LoRA秩
        steps: 训练步数
        status: 状态组件
    """
    # 初始化状态
    initial_status = f"""
======== 离线训练初始化 ========
项目: {basename}
模型大小: {model_size}
分辨率: {resolution}
LoRA秩: {rank}
训练步数: {steps}
==============================
"""
    if hasattr(status, 'update'):
        status.update(value=initial_status)
    
    # 检查数据集路径
    dataset_path = check_dataset_location(basename)
    if not dataset_path:
        error_msg = f"错误: 未找到数据集 '{basename}'\n请确保数据位于 'train_date/{basename}' 或 '{basename}_raw' 目录"
        if hasattr(status, 'update'):
            status.update(value=initial_status + "\n" + error_msg)
        logger.error(error_msg)
        return initial_status + "\n" + error_msg
        
    update_status = initial_status + f"\n找到数据集: {dataset_path}\n\n正在准备训练环境...\n"
    if hasattr(status, 'update'):
        status.update(value=update_status)
    
    # 查找模型文件
    model_pattern = "ltxv-13b" if model_size == "13B" else "ltx-video-2b"
    models_dir = os.path.join(PROJECT_DIR, "models")
    model_files = list(Path(models_dir).glob(f"*{model_pattern}*.safetensors"))
    
    if not model_files:
        error_msg = f"错误: 在models目录中未找到{model_size}模型文件"
        if hasattr(status, 'update'):
            status.update(value=update_status + "\n" + error_msg)
        logger.error(error_msg)
        return update_status + "\n" + error_msg
    
    model_path = model_files[0]
    logger.info(f"使用模型: {model_path}")
    update_status += f"使用模型: {model_path}\n\n开始训练过程..."
    
    if hasattr(status, 'update'):
        status.update(value=update_status)
    
    # 获取增强版训练脚本路径
    enhanced_script_path = os.path.join(PROJECT_DIR, "scripts", "enhanced_offline_train.py")
    
    # 组装命令
    cmd = [
        sys.executable,
        enhanced_script_path,
        basename,
        "--model-size", model_size,
        "--resolution", resolution,
        "--rank", str(rank),
        "--steps", str(steps)
    ]
    
    # 打印命令
    cmd_line = " ".join(cmd)
    logger.info(f"执行命令: {cmd_line}")
    update_status += f"\n\n执行命令: {cmd_line}\n\n== 训练日志开始 ==\n"
    
    if hasattr(status, 'update'):
        status.update(value=update_status)
    
    # 运行命令并捕获输出
    def run_and_update():
        nonlocal update_status
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,  # 使用bytes模式而不是text模式
            bufsize=1
            # 移除universal_newlines=True以避免默认编码问题
        )
        
        output_lines = []
        for line in process.stdout:
            # 使用utf-8解码，忽略无法解码的字符
            try:
                decoded_line = line.decode('utf-8', errors='replace').strip()
                print(decoded_line)  # 在控制台显示
                output_lines.append(decoded_line)
            except Exception as e:
                # 如果出错，转换为安全的字符串
                safe_line = str(line).replace('\\', '/').strip()
                print(f"[解码错误] {safe_line}")
                output_lines.append(f"[解码错误] {safe_line}")
            
            # 更新UI状态 - 保持最新的25行
            if len(output_lines) <= 30:
                current_status = update_status + "\n".join(output_lines)
            else:
                # 保留开头和最新的日志
                current_status = update_status + "...\n" + "\n".join(output_lines[-25:])
            
            if hasattr(status, 'update'):
                status.update(value=current_status)
        
        process.wait()
        
        if process.returncode == 0:
            final_status = update_status + "\n".join(output_lines) + "\n\n== 训练日志结束 ==\n\n✅ 训练成功完成!\n"
            final_status += f"结果保存在: outputs/{basename}_offline_training/lora_weights/\n"
            final_status += "包含以下文件:\n"
            final_status += "- adapter_model.safetensors (LoRA权重)\n"
            final_status += "- adapter_config.json (LoRA配置)\n"
        else:
            final_status = update_status + "\n".join(output_lines) + "\n\n== 训练日志结束 ==\n\n❌ 训练过程中出错!\n"
            final_status += f"退出码: {process.returncode}\n"
            
        if hasattr(status, 'update'):
            status.update(value=final_status)
            
    # 使用线程执行训练
    thread = threading.Thread(target=run_and_update)
    thread.daemon = True
    thread.start()
    
    # 返回初始状态 - 由线程更新UI
    return update_status + "训练已启动，正在生成日志..."

def find_ltx_model(model_name_pattern="ltxv-13b"):
    """查找LTX模型文件
    
    Args:
        model_name_pattern: 模型名称模式，如ltxv-13b
        
    Returns:
        找到的模型路径或None
    """
    for path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(path):
            for file in os.listdir(path):
                if model_name_pattern in file.lower() and file.endswith(".safetensors"):
                    full_path = os.path.join(path, file)
                    logger.info(f"找到LTX模型: {full_path}")
                    return full_path
    
    logger.warning(f"未找到模型: {model_name_pattern}")
    return None

def check_models():
    """检查models目录中的模型文件"""
    model_status = []
    
    # 搜索多个路径中的模型
    ltx_2b_model = find_ltx_model("ltx-video-2b")
    ltx_13b_model = find_ltx_model("ltxv-13b")
        
    # 检查LTX 2B模型
    if ltx_2b_model:
        model_status.append(f"✅ 2B模型: {os.path.basename(ltx_2b_model)}")
    else:
        model_status.append("❌ 未找到2B模型")
    
    # 检查LTX 13B模型
    if ltx_13b_model:
        model_status.append(f"✅ 13B模型: {os.path.basename(ltx_13b_model)}")
    else:
        model_status.append("❌ 未找到13B模型")
        
    return model_status

def run_preprocessing(dataset, resolution, id_token, decode_videos, status):
    """运行数据预处理
    
    Args:
        dataset: 数据集名称或路径
        resolution: 分辨率
        id_token: ID标记
        decode_videos: 是否解码视频
        status: 状态组件
    """
    initial_status = f"""====== 数据预处理 ======
数据集: {dataset}
分辨率: {resolution}
ID标记: {id_token if id_token else '无'}
解码视频: {'是' if decode_videos else '否'}
========================
"""
    
    if hasattr(status, 'update'):
        status.update(value=initial_status)
    
    # 检查数据集路径
    dataset_path = check_dataset_location(dataset)
    if not dataset_path:
        error_msg = f"错误: 未找到数据集 '{dataset}'"
        logger.error(error_msg)
        if hasattr(status, 'update'):
            status.update(value=initial_status + "\n" + error_msg)
        return initial_status + "\n" + error_msg
    
    # 准备命令
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
    
    # 从配置文件加载预处理参数
    config_path = os.path.join(CONFIG_DIR, "default_config.yaml")
    if os.path.exists(config_path):
        config = load_config(config_path)
        if config and 'preprocessing' in config:
            preproc_config = config['preprocessing']
            
            # 是否跳过已处理的场景
            if preproc_config.get('skip_existing_scenes', True):
                cmd.append("--skip-existing")
    
    # 更新状态
    update_msg = initial_status + f"\n数据集路径: {dataset_path}\n\n准备运行预处理命令...\n"
    if hasattr(status, 'update'):
        status.update(value=update_msg)
    
    # 运行命令
    logger.info(f"运行预处理命令: {' '.join(cmd)}")
    try:
        returncode, output = run_command(cmd, status=status)
        if returncode == 0:
            success_msg = "\n\n✅ 预处理成功完成!"
            if hasattr(status, 'update'):
                status.update(value=output + success_msg)
            return output + success_msg
        else:
            error_msg = "\n\n❌ 预处理失败!"
            if hasattr(status, 'update'):
                status.update(value=output + error_msg)
            return output + error_msg
    except Exception as e:
        error_msg = f"\n\n❌ 预处理出错: {str(e)}"
        if hasattr(status, 'update'):
            status.update(value=update_msg + error_msg)
        return update_msg + error_msg

def convert_to_comfyui(input_path, output_path, status):
    """转换为ComfyUI格式
    
    Args:
        input_path: 输入模型路径
        output_path: 输出路径
        status: 状态组件
    """
    initial_status = f"""====== 转换为ComfyUI格式 ======
输入路径: {input_path}
输出路径: {output_path if output_path else '自动生成'}
========================
"""
    
    if hasattr(status, 'update'):
        status.update(value=initial_status)
    
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        error_msg = f"错误: 输入路径不存在 '{input_path}'"
        logger.error(error_msg)
        if hasattr(status, 'update'):
            status.update(value=initial_status + "\n" + error_msg)
        return initial_status + "\n" + error_msg
    
    # 准备命令
    cmd = [
        sys.executable,
        os.path.join(SCRIPTS_DIR, "convert_checkpoint.py"),
        input_path,
        "--to-comfy"
    ]
    
    if output_path:
        cmd.extend(["--output_path", output_path])
    
    # 更新状态
    update_msg = initial_status + "\n准备转换...\n"
    if hasattr(status, 'update'):
        status.update(value=update_msg)
    
    # 运行命令
    try:
        returncode, output = run_command(cmd, status=status)
        if returncode == 0:
            success_msg = "\n\n✅ 转换成功完成!"
            if hasattr(status, 'update'):
                status.update(value=output + success_msg)
            return output + success_msg
        else:
            error_msg = "\n\n❌ 转换失败!"
            if hasattr(status, 'update'):
                status.update(value=output + error_msg)
            return output + error_msg
    except Exception as e:
        error_msg = f"\n\n❌ 转换出错: {str(e)}"
        if hasattr(status, 'update'):
            status.update(value=update_msg + error_msg)
        return update_msg + error_msg

def main():
    """创建Gradio UI界面"""
    with gr.Blocks(title="LTX-Video-Trainer 离线训练工具") as app:
        gr.Markdown("# 🚀 LTX-Video-Trainer 离线训练工具")
        gr.Markdown("### 完全离线模式 - 基于本地模型文件生成LoRA权重")
        
        # 显示模型状态
        gr.Markdown("## 模型状态")
        model_status = check_models()
        gr.Markdown("\n".join(model_status))
        
        with gr.Tabs():
            # 离线训练模式标签页
            with gr.TabItem("本地离线训练"):
                gr.Markdown("### 🔥 完全离线模式 - 不需要下载任何资源")
                gr.Markdown("该模式使用本地模型文件直接生成LoRA权重，简化训练流程并避免任何网络请求")
                
                with gr.Row():
                    with gr.Column():
                        offline_basename = gr.Textbox(
                            label="项目名称", 
                            placeholder="输入项目名称，如APT，数据集应放在train_date/APT目录中"
                        )
                        gr.Markdown("支持的数据集位置：**train_date/{项目名}** 或 **{项目名}_raw**")
                        offline_model_size = gr.Radio(
                            label="模型大小", 
                            choices=["2B", "13B"], 
                            value="2B",
                            info="2B模型需要较少的显存，13B模型需要更多显存但质量更高"
                        )
                        offline_resolution = gr.Dropdown(
                            label="分辨率", 
                            choices=RESOLUTIONS + ["576x1024x41"],
                            value="768x768x25",
                            info="格式为宽x高x帧数，帧数较少训练更快"
                        )
                        offline_rank = gr.Slider(
                            label="LoRA秩 (Rank)", 
                            minimum=1, 
                            maximum=128, 
                            value=32,
                            info="值越大，适应性越强但需要更多显存"
                        )
                        offline_steps = gr.Slider(
                            label="训练步数", 
                            minimum=5, 
                            maximum=200, 
                            value=50,
                            info="步数越多，训练时间越长，但效果可能更好"
                        )
                        offline_button = gr.Button(
                            "开始本地离线训练", 
                            variant="primary"
                        )
                    
                    with gr.Column():
                        offline_status = gr.Textbox(
                            label="训练日志", 
                            lines=25,
                            max_lines=30,
                            autoscroll=True
                        )
                
                offline_button.click(
                    fn=run_offline_training,
                    inputs=[
                        offline_basename, 
                        offline_model_size, 
                        offline_resolution, 
                        offline_rank, 
                        offline_steps, 
                        offline_status
                    ],
                    outputs=offline_status
                )
        
        # 页脚信息
        gr.Markdown("---")
        gr.Markdown("### 📝 使用说明")
        gr.Markdown("""
        1. 确保您的数据集已放在正确位置: `train_date/{项目名}` 或 `{项目名}_raw`
        2. 选择合适的模型大小、分辨率和训练参数
        3. 点击"开始本地离线训练"按钮
        4. 训练日志会实时显示在右侧文本框和终端中
        5. 训练完成后，LoRA文件将保存在 `outputs/{项目名}_offline_training/lora_weights` 目录
        """)
        
    # 启动UI
    app.launch(server_name="127.0.0.1", server_port=7860)

if __name__ == "__main__":
    main()
