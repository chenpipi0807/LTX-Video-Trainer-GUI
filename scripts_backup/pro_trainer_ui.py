#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTX-Video-Trainer 专业版训练界面
整合了所有高级训练参数控制功能
"""

import os
import sys
import json
import logging
import threading
import subprocess
from pathlib import Path
import gradio as gr

# 获取项目根目录
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
sys.path.append(os.path.join(PROJECT_DIR, "scripts"))

# 导入高级训练参数模块
from enhanced_training_params import *
from advanced_training_ui import *

# 可能的LTX模型路径
POSSIBLE_MODEL_PATHS = [
    # 项目内部路径
    os.path.join(PROJECT_DIR, "models"),
    # ComfyUI路径
    r"C:\NEWCOMFYUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints",
]

# 分辨率选项
RESOLUTIONS = [
    "512x512x25", "576x576x25", "640x640x25", "704x704x25", "768x768x25",
    "512x512x49", "576x576x49", "640x640x49", "704x704x49", "768x768x49",
    "576x1024x41", "1024x576x41"
]

# 设置环境变量强制离线模式
os.environ["PYTHONIOENCODING"] = "utf8"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('LTX-Pro-Trainer')

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

def find_ltx_model(model_name_pattern="ltxv-13b"):
    """查找LTX模型文件"""
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
    
    # 搜索LTX 2B模型
    ltx_2b_model = find_ltx_model("ltx-video-2b")
    if ltx_2b_model:
        model_status.append(f"✅ 2B模型: {os.path.basename(ltx_2b_model)}")
    else:
        model_status.append("❌ 未找到2B模型")
    
    # 搜索LTX 13B模型
    ltx_13b_model = find_ltx_model("ltxv-13b")
    if ltx_13b_model:
        model_status.append(f"✅ 13B模型: {os.path.basename(ltx_13b_model)}")
    else:
        model_status.append("⚠️ 未找到13B模型 (可选)")
    
    return model_status

def run_advanced_training(basename, model_size, resolution, config, status):
    """
    运行高级训练流程
    
    Args:
        basename: 项目名称
        model_size: 模型大小 (2B 或 13B)
        resolution: 分辨率
        config: 训练配置
        status: 状态组件
    """
    # 初始化状态
    initial_status = f"""
======== 高级训练初始化 ========
项目: {basename}
模型大小: {model_size}
分辨率: {resolution}
预设: {config.get('_preset_name', '自定义')}
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
    model_path = find_ltx_model(model_pattern)
    
    if not model_path:
        error_msg = f"错误: 未找到{model_size}模型文件"
        if hasattr(status, 'update'):
            status.update(value=update_status + "\n" + error_msg)
        logger.error(error_msg)
        return update_status + "\n" + error_msg
    
    # 更新配置中的模型路径和分辨率
    config["model"]["model_source"] = str(model_path)
    update_video_dims_from_resolution(config, resolution)
    
    # 设置输出目录
    output_dir = os.path.join(PROJECT_DIR, "outputs", f"{basename}_advanced_training")
    config["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置预处理数据根目录
    preprocessed_data_root = os.path.join(PROJECT_DIR, f"{basename}_scenes", ".precomputed")
    config["data"]["preprocessed_data_root"] = preprocessed_data_root
    
    # 保存配置文件
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    # 准备训练脚本
    logger.info(f"使用模型: {model_path}")
    logger.info(f"配置已保存到: {config_path}")
    update_status += f"使用模型: {model_path}\n配置已保存到: {config_path}\n\n执行训练..."
    
    if hasattr(status, 'update'):
        status.update(value=update_status)
    
    # 获取增强版训练脚本路径
    script_path = os.path.join(PROJECT_DIR, "scripts", "enhanced_offline_train.py")
    
    # 组装命令
    cmd = [
        sys.executable,
        script_path,
        basename,
        "--model-size", model_size,
        "--resolution", resolution,
        "--rank", str(config["lora"]["rank"]),
        "--steps", str(config["optimization"]["steps"]),
        "--config", config_path
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
            final_status += f"结果保存在: {output_dir}/lora_weights/\n"
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

def main():
    """创建专业版训练UI界面"""
    # 确保配置目录存在
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # 创建默认配置文件（如果不存在）
    create_default_config_if_missing()
    
    with gr.Blocks(title="LTX-Video-Trainer 专业版训练界面") as app:
        gr.Markdown("# 🚀 LTX-Video-Trainer 专业版训练界面")
        gr.Markdown("### 为LTX视频模型提供高度可定制的训练控制")
        
        # 显示模型状态
        gr.Markdown("## 模型状态")
        model_status = check_models()
        gr.Markdown("\n".join(model_status))
        
        with gr.Tabs():
            # 高级训练模式标签页
            with gr.TabItem("高级训练"):
                gr.Markdown("### 🧪 高级训练控制 - 完全自定义训练参数")
                
                # 创建基本配置区
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 基本设置")
                        basename = gr.Textbox(
                            label="项目名称", 
                            placeholder="输入项目名称，如APT，数据集应放在train_date/APT目录中",
                            info="数据集位置：train_date/{项目名} 或 {项目名}_raw"
                        )
                        model_size = gr.Radio(
                            label="模型大小", 
                            choices=["2B", "13B"], 
                            value="2B",
                            info="2B模型需要较少的显存，13B模型需要更多显存但质量更高"
                        )
                        resolution = gr.Dropdown(
                            label="分辨率", 
                            choices=RESOLUTIONS,
                            value="768x768x25",
                            info="格式为宽x高x帧数，帧数较少训练更快"
                        )
                        
                        # 创建预设选择器
                        preset_ui = create_preset_selector_ui()
                        
                        # 创建训练按钮
                        train_button = gr.Button("开始高级训练", variant="primary")
                    
                    # 右侧显示训练状态
                    with gr.Column(scale=1):
                        train_status = gr.Textbox(
                            label="训练日志", 
                            lines=30,
                            max_lines=40,
                            autoscroll=True
                        )
                
                # 创建高级参数控制区
                with gr.Tabs():
                    with gr.TabItem("LoRA参数"):
                        lora_ui = create_lora_params_ui()
                        
                    with gr.TabItem("优化参数"):
                        optimization_ui = create_optimization_params_ui()
                        
                    with gr.TabItem("加速参数"):
                        acceleration_ui = create_acceleration_params_ui()
                        
                    with gr.TabItem("数据参数"):
                        data_ui = create_data_params_ui()
                        
                    with gr.TabItem("验证参数"):
                        validation_ui = create_validation_params_ui()
                        
                    with gr.TabItem("高级选项"):
                        advanced_ui = create_advanced_options_ui()
                
                # 合并所有UI组件
                all_ui_components = {}
                all_ui_components.update(lora_ui)
                all_ui_components.update(optimization_ui)
                all_ui_components.update(acceleration_ui)
                all_ui_components.update(data_ui)
                all_ui_components.update(validation_ui)
                all_ui_components.update(advanced_ui)
                all_ui_components.update(preset_ui)
                
                # 配置预设按钮点击事件
                def apply_preset_handler(preset_name):
                    if not preset_name:
                        return [gr.update() for _ in range(len(all_ui_components))]
                    
                    try:
                        # 记录日志
                        logger.info(f"已应用预设配置: {preset_name}")
                        
                        # 获取当前配置
                        current_config = collect_advanced_params(all_ui_components)
                        
                        # 应用预设
                        new_config = apply_preset(current_config, preset_name)
                        new_config["_preset_name"] = preset_name
                        
                        # 设置UI组件的值 - 直接返回更新对象列表而不是实际组件
                        return set_ui_values_from_config(all_ui_components, new_config)
                    except Exception as e:
                        logger.error(f"应用预设出错: {str(e)}")
                        # 出错时返回空更新
                        return [gr.update() for _ in range(len(all_ui_components))]
                
                # 连接预设按钮
                preset_ui["apply_preset_btn"].click(
                    fn=apply_preset_handler,
                    inputs=[preset_ui["preset_selector"]],
                    outputs=list(all_ui_components.values())
                )
                
                # 训练按钮事件处理
                def train_handler(
                    basename, model_size, resolution, 
                    **kwargs
                ):
                    # 收集所有参数
                    config = collect_advanced_params(all_ui_components)
                    
                    # 启动训练
                    return run_advanced_training(
                        basename, model_size, resolution, 
                        config, train_status
                    )
                
                # 连接训练按钮
                train_button.click(
                    fn=train_handler,
                    inputs=[basename, model_size, resolution],
                    outputs=[train_status]
                )
            
            # 标准训练模式标签页
            with gr.TabItem("标准训练"):
                gr.Markdown("### 🚀 标准训练模式 - 简化的训练流程")
                
                with gr.Row():
                    with gr.Column():
                        std_basename = gr.Textbox(
                            label="项目名称", 
                            placeholder="输入项目名称，如APT，数据集应放在train_date/APT目录中"
                        )
                        gr.Markdown("支持的数据集位置：**train_date/{项目名}** 或 **{项目名}_raw**")
                        std_model_size = gr.Radio(
                            label="模型大小", 
                            choices=["2B", "13B"], 
                            value="2B",
                            info="2B模型需要较少的显存，13B模型需要更多显存但质量更高"
                        )
                        std_resolution = gr.Dropdown(
                            label="分辨率", 
                            choices=RESOLUTIONS,
                            value="768x768x25",
                            info="格式为宽x高x帧数，帧数较少训练更快"
                        )
                        std_rank = gr.Slider(
                            label="LoRA秩 (Rank)", 
                            minimum=1, 
                            maximum=128, 
                            value=32,
                            info="值越大，适应性越强但需要更多显存"
                        )
                        std_steps = gr.Slider(
                            label="训练步数", 
                            minimum=5, 
                            maximum=200, 
                            value=50,
                            info="步数越多，训练时间越长，但效果可能更好"
                        )
                        std_train_button = gr.Button(
                            "开始标准训练", 
                            variant="primary"
                        )
                    
                    with gr.Column():
                        std_status = gr.Textbox(
                            label="训练日志", 
                            lines=25,
                            max_lines=30,
                            autoscroll=True
                        )
                
                # 连接标准训练按钮
                def std_train_handler(basename, model_size, resolution, rank, steps):
                    # 创建基本配置
                    model_pattern = "ltxv-13b" if model_size == "13B" else "ltx-video-2b"
                    model_path = find_ltx_model(model_pattern)
                    
                    if not model_path:
                        return f"错误: 未找到{model_size}模型文件"
                    
                    # 创建标准训练配置
                    config = create_training_config_from_params(
                        basename, model_path, resolution, rank, steps
                    )
                    config["_preset_name"] = "标准训练"
                    
                    # 启动训练
                    return run_advanced_training(
                        basename, model_size, resolution, 
                        config, std_status
                    )
                
                std_train_button.click(
                    fn=std_train_handler,
                    inputs=[std_basename, std_model_size, std_resolution, std_rank, std_steps],
                    outputs=[std_status]
                )
        
        # 页脚信息
        gr.Markdown("---")
        gr.Markdown("### 📝 使用说明")
        gr.Markdown("""
        #### 高级训练模式使用指南:
        
        1. **基本设置**:
           - 选择项目名称（对应数据集名称）
           - 选择模型大小和分辨率
           - 可以选择预设配置以快速设置参数
        
        2. **参数调整**:
           - **LoRA参数**: 调整秩、Alpha和Dropout等
           - **优化参数**: 控制学习率、训练步数和优化器类型
           - **加速参数**: 设置混合精度和内存优化选项
           - **数据参数**: 配置数据增强和加载选项
           - **验证参数**: 设置生成验证视频的参数
           - **高级选项**: 调整其他高级训练选项
        
        3. **训练过程**:
           - 点击"开始高级训练"按钮开始训练
           - 训练日志会实时显示在右侧文本框
           - 训练完成后，结果保存在`outputs/{项目名}_advanced_training/lora_weights`目录
        
        #### 提示:
        - 对于初学者，建议使用"标准训练"标签页
        - 高级用户可以通过"高级训练"标签页调整所有参数
        - 显存不足时可以应用"低显存模式"预设
        """)
    
    # 启动UI
    app.launch(server_name="127.0.0.1", server_port=7862)

if __name__ == "__main__":
    main()
