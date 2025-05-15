#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTX-Video-Trainer 极简Gradio界面
适应不同环境配置的简化版UI
"""

import os
import sys
import subprocess
from pathlib import Path

# 检查gradio是否可用
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("警告: Gradio未安装，将使用命令行界面")

# 检查torch是否可用
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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

def run_command(cmd, status_output=None):
    """运行命令并显示输出"""
    cmd_str = " ".join(cmd)
    
    if status_output:
        status_output.update(f"运行命令:\n{cmd_str}\n\n请等待...")
    else:
        print(f"\n执行命令: {cmd_str}\n")
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        for line in process.stdout:
            line = line.strip()
            output_lines.append(line)
            
            if len(output_lines) > 50:
                output_lines = output_lines[-50:]  # 保留最后50行
            
            if status_output:
                status_output.update(f"运行命令:\n{cmd_str}\n\n" + "\n".join(output_lines))
            else:
                print(line)
        
        process.wait()
        
        result_msg = ""
        if process.returncode == 0:
            result_msg = "\n命令执行成功!"
        else:
            result_msg = f"\n命令执行失败，返回代码 {process.returncode}"
        
        if status_output:
            status_output.update(status_output.value + result_msg)
        else:
            print(result_msg)
            
        return True if process.returncode == 0 else False
    
    except Exception as e:
        error_msg = f"\n执行出错: {str(e)}"
        if status_output:
            status_output.update(status_output.value + error_msg)
        else:
            print(error_msg)
        return False

def run_preprocessing(args):
    """运行数据预处理"""
    cmd = [
        sys.executable,
        os.path.join(SCRIPTS_DIR, "preprocess_dataset.py"),
        args["dataset_path"],
        "--resolution-buckets", args["resolution"]
    ]
    
    if args.get("id_token"):
        cmd.extend(["--id-token", args["id_token"]])
    
    if args.get("decode_videos"):
        cmd.append("--decode-videos")
    
    return run_command(cmd, args.get("status_output"))

def run_training(args):
    """运行训练脚本"""
    cmd = [
        sys.executable,
        os.path.join(SCRIPTS_DIR, "train.py"),
        args["config_path"]
    ]
    
    return run_command(cmd, args.get("status_output"))

def run_pipeline(args):
    """运行完整流水线"""
    cmd = [
        sys.executable,
        os.path.join(SCRIPTS_DIR, "run_pipeline.py"),
        args["basename"],
        "--resolution-buckets", args["resolution"],
        "--config-template", args["config_template"],
        "--rank", str(args["rank"])
    ]
    
    return run_command(cmd, args.get("status_output"))

def convert_to_comfyui(args):
    """转换为ComfyUI格式"""
    cmd = [
        sys.executable,
        os.path.join(SCRIPTS_DIR, "convert_checkpoint.py"),
        args["input_path"],
        "--to-comfy"
    ]
    
    if args.get("output_path"):
        cmd.extend(["--output_path", args["output_path"]])
    
    return run_command(cmd, args.get("status_output"))

def create_gradio_ui():
    """创建Gradio界面"""
    with gr.Blocks(title="LTX-Video训练器") as app:
        gr.Markdown("# 🎬 LTX-Video训练器")
        gr.Markdown("### 视频模型训练界面")
        
        # GPU信息
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gr.Markdown(f"**GPU: {gpu_name} | 显存: {gpu_memory:.1f} GB**")
        else:
            gr.Markdown("**⚠️ 未检测到GPU或PyTorch。训练需要NVIDIA GPU支持。**")
        
        with gr.Tabs():
            # 完整流水线标签页
            with gr.Tab("一键训练流水线"):
                gr.Markdown("### 🚀 从原始视频到训练模型的全流程")
                
                pipeline_basename = gr.Textbox(label="项目名称", placeholder="例如: my_effect")
                pipeline_resolution = gr.Dropdown(label="分辨率", choices=RESOLUTIONS, value="768x768x49")
                pipeline_config = gr.Dropdown(label="配置模板", choices=list(CONFIG_FILES.keys()))
                pipeline_rank = gr.Slider(label="LoRA秩 (Rank)", minimum=1, maximum=128, value=32)
                pipeline_status = gr.Textbox(label="状态", interactive=False, lines=15)
                pipeline_button = gr.Button("开始一键训练", variant="primary")
                
                def run_pipeline_ui(basename, resolution, config_template, rank):
                    if not basename:
                        return "错误: 项目名称不能为空"
                    
                    args = {
                        "basename": basename,
                        "resolution": resolution,
                        "config_template": CONFIG_FILES.get(config_template),
                        "rank": rank,
                        "status_output": pipeline_status
                    }
                    
                    run_pipeline(args)
                    return pipeline_status.value
                
                pipeline_button.click(
                    run_pipeline_ui,
                    inputs=[pipeline_basename, pipeline_resolution, pipeline_config, pipeline_rank],
                    outputs=pipeline_status
                )
            
            # 数据预处理标签页
            with gr.Tab("数据预处理"):
                gr.Markdown("### 🔄 准备数据集")
                
                preprocess_dataset = gr.Textbox(label="数据集路径", placeholder="数据集目录或元数据文件路径")
                preprocess_resolution = gr.Dropdown(label="分辨率", choices=RESOLUTIONS, value="768x768x25")
                preprocess_id_token = gr.Textbox(label="ID标记 (LoRA触发词)", placeholder="例如: <特效>")
                preprocess_decode = gr.Checkbox(label="解码视频进行验证", value=True)
                preprocess_status = gr.Textbox(label="状态", interactive=False, lines=15)
                preprocess_button = gr.Button("开始预处理", variant="primary")
                
                def run_preprocess_ui(dataset_path, resolution, id_token, decode_videos):
                    if not dataset_path:
                        return "错误: 数据集路径不能为空"
                    
                    args = {
                        "dataset_path": dataset_path,
                        "resolution": resolution,
                        "id_token": id_token,
                        "decode_videos": decode_videos,
                        "status_output": preprocess_status
                    }
                    
                    run_preprocessing(args)
                    return preprocess_status.value
                
                preprocess_button.click(
                    run_preprocess_ui,
                    inputs=[preprocess_dataset, preprocess_resolution, preprocess_id_token, preprocess_decode],
                    outputs=preprocess_status
                )
            
            # 模型训练标签页
            with gr.Tab("模型训练"):
                gr.Markdown("### 🚂 训练模型")
                
                train_config = gr.Dropdown(label="训练配置文件", choices=list(CONFIG_FILES.keys()))
                train_status = gr.Textbox(label="状态", interactive=False, lines=15)
                train_button = gr.Button("开始训练", variant="primary")
                
                def run_train_ui(config_name):
                    if not config_name:
                        return "错误: 请选择配置文件"
                    
                    args = {
                        "config_path": CONFIG_FILES.get(config_name),
                        "status_output": train_status
                    }
                    
                    run_training(args)
                    return train_status.value
                
                train_button.click(
                    run_train_ui,
                    inputs=[train_config],
                    outputs=train_status
                )
            
            # 转换标签页
            with gr.Tab("转换为ComfyUI格式"):
                gr.Markdown("### 🔄 转换模型格式")
                
                convert_input = gr.Textbox(label="输入模型路径", placeholder="训练好的模型权重路径 (.safetensors)")
                convert_output = gr.Textbox(label="输出路径 (可选)", placeholder="留空则自动命名")
                convert_status = gr.Textbox(label="状态", interactive=False, lines=10)
                convert_button = gr.Button("转换为ComfyUI格式", variant="primary")
                
                def run_convert_ui(input_path, output_path):
                    if not input_path:
                        return "错误: 输入路径不能为空"
                    
                    args = {
                        "input_path": input_path,
                        "output_path": output_path,
                        "status_output": convert_status
                    }
                    
                    convert_to_comfyui(args)
                    return convert_status.value
                
                convert_button.click(
                    run_convert_ui,
                    inputs=[convert_input, convert_output],
                    outputs=convert_status
                )
            
            # 帮助标签页
            with gr.Tab("帮助"):
                gr.Markdown("""
                # LTX-Video-Trainer 使用帮助
                
                ## 训练数据要求
                
                - **数量**: 通常5-50个视频效果的样本即可
                - **长度**: 推荐5-15秒的短视频片段
                - **质量**: 高质量、清晰的视频效果样本
                - **内容**: 集中展示您想要训练的特效
                
                ## 硬件要求
                
                - **GPU**: 至少24GB显存的NVIDIA GPU (用于2B模型)
                - **CPU**: 多核处理器
                - **内存**: 至少16GB RAM
                - **存储**: 至少50GB可用空间
                
                ## 快速开始指南
                
                ### 使用一键流水线:
                
                1. 创建名为`项目名_raw`的文件夹
                2. 将原始视频放入该文件夹
                3. 在界面中填写项目名称(不含"_raw"后缀)
                4. 选择分辨率和配置模板
                5. 点击"开始一键训练"
                
                ### 使用自定义工作流:
                
                1. **数据预处理**:
                   - 提供数据集路径(视频文件夹或元数据文件)
                   - 选择分辨率
                   - 设置LoRA触发词(可选)
                   - 点击"开始预处理"
                
                2. **模型训练**:
                   - 选择配置文件
                   - 点击"开始训练"
                
                3. **转换格式**:
                   - 提供训练好的权重文件路径
                   - 点击"转换为ComfyUI格式"
                
                ## 分辨率选择指南
                
                分辨率格式为"宽x高x帧数":
                
                - **512x512x25**: 基础分辨率，适合低显存
                - **768x768x25**: 中等分辨率，更好的细节
                - **768x768x49**: 更多帧数，捕捉更多动态
                - **1024x576x41**: 宽屏格式，高清细节
                """)
        
        gr.Markdown("*感谢使用LTX-Video训练器*")
    
    return app

def run_cli():
    """运行命令行界面"""
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
            print("\n===== 一键训练流水线 =====")
            basename = input("\n项目名称 (例如: my_effect): ")
            
            print("\n可用分辨率:")
            for i, res in enumerate(RESOLUTIONS, 1):
                print(f"{i}. {res}")
            
            res_choice = int(input("\n选择分辨率 (输入编号): "))
            resolution = RESOLUTIONS[res_choice - 1]
            
            print("\n可用配置模板:")
            config_names = list(CONFIG_FILES.keys())
            for i, name in enumerate(config_names, 1):
                print(f"{i}. {name}")
            
            config_choice = int(input("\n选择配置模板 (输入编号): "))
            config_template = CONFIG_FILES[config_names[config_choice - 1]]
            
            rank = int(input("\nLoRA秩 (1-128，推荐32): ") or "32")
            
            args = {
                "basename": basename,
                "resolution": resolution,
                "config_template": config_template,
                "rank": rank
            }
            
            run_pipeline(args)
        
        elif choice == "2":
            print("\n===== 数据预处理 =====")
            dataset_path = input("\n数据集路径 (文件夹或元数据文件): ")
            
            print("\n可用分辨率:")
            for i, res in enumerate(RESOLUTIONS, 1):
                print(f"{i}. {res}")
            
            res_choice = int(input("\n选择分辨率 (输入编号): "))
            resolution = RESOLUTIONS[res_choice - 1]
            
            id_token = input("\nLoRA触发词 (可选，按Enter跳过): ")
            decode_videos = input("\n解码视频进行验证? (y/n): ").lower() == "y"
            
            args = {
                "dataset_path": dataset_path,
                "resolution": resolution,
                "id_token": id_token,
                "decode_videos": decode_videos
            }
            
            run_preprocessing(args)
        
        elif choice == "3":
            print("\n===== 模型训练 =====")
            
            print("\n可用配置:")
            config_names = list(CONFIG_FILES.keys())
            for i, name in enumerate(config_names, 1):
                print(f"{i}. {name}")
            
            config_choice = int(input("\n选择配置 (输入编号): "))
            config_path = CONFIG_FILES[config_names[config_choice - 1]]
            
            args = {
                "config_path": config_path
            }
            
            run_training(args)
        
        elif choice == "4":
            print("\n===== 转换为ComfyUI格式 =====")
            
            input_path = input("\n输入模型路径 (.safetensors): ")
            output_path = input("\n输出路径 (可选，按Enter跳过): ")
            
            args = {
                "input_path": input_path,
                "output_path": output_path
            }
            
            convert_to_comfyui(args)
        
        elif choice == "5":
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
        
        elif choice == "0":
            print("\n感谢使用LTX-Video训练器!")
            break
        
        else:
            print("\n无效的选择，请重试。")

if __name__ == "__main__":
    # 尝试使用Gradio界面，如果不可用则回退到命令行
    if GRADIO_AVAILABLE:
        try:
            app = create_gradio_ui()
            print("正在启动Gradio界面...")
            app.launch(share=False)
        except Exception as e:
            print(f"启动Gradio界面失败: {str(e)}")
            print("回退到命令行界面")
            run_cli()
    else:
        print("未检测到Gradio，使用命令行界面...")
        run_cli()
