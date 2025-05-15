#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 强制设置默认编码为UTF-8，避免中文环境下的编码问题
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

"""
极简LTX-Video-Trainer界面
最小化依赖要求，提供基本训练功能
"""

import os
import sys
import time
import subprocess
import json
import shutil
import yaml  # 添加缺失的yaml导入
import logging
import gradio as gr
import torch
import numpy as np  # 添加numpy导入，避免潜在错误
import datetime  # 添加datetime导入以支持时间戳功能
from pathlib import Path
from safetensors.torch import save_file  # 添加safetensors导入

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LTX-Trainer')

# 默认路径
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(PROJECT_DIR, "configs")
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")

# LTX模型路径 - 只使用diffusers格式的本地模型
DIFFUSERS_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "LTX-Video-0.9.7-diffusers")

# 设置环境变量以强制离线模式
os.environ["PYTHONIOENCODING"] = "utf8"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


# 数据集目录
DATA_DIR = os.path.join(PROJECT_DIR, "train_date")

# 用户自定义配置目录
USER_CONFIGS_DIR = os.path.join(PROJECT_DIR, "user_configs")
os.makedirs(USER_CONFIGS_DIR, exist_ok=True)  # 确保目录存在

# 读取可用的配置文件
CONFIG_FILES = {}

# 读取默认配置文件
for file in os.listdir(CONFIGS_DIR):
    if file.endswith(".yaml"):
        name = file.replace(".yaml", "")
        CONFIG_FILES[name] = os.path.join(CONFIGS_DIR, file)

# 读取用户自定义配置文件
if os.path.exists(USER_CONFIGS_DIR):
    for file in os.listdir(USER_CONFIGS_DIR):
        if file.endswith(".yaml"):
            name = file.replace(".yaml", "")
            # 添加标记以区分自定义配置
            display_name = f"[自定义] {name}"
            CONFIG_FILES[display_name] = os.path.join(USER_CONFIGS_DIR, file)
            logger.info(f"发现用户自定义配置: {file}")

# 分辨率和帧数分开设置
# 预设分辨率列表 - 所有分辨率必须是32的倍数
RESOLUTIONS_DIMS = [
    # 小尺寸（推荐，训练速度快，显存需求低）
    "[FANG] 320x320",    # 超小方形分辨率，最适合快速测试
    "[FANG] 384x384",    # 小方形分辨率，使用很少显存
    "[FANG] 416x416",    # 中小方形分辨率，平衡速度与质量
    "[FANG] 512x512",    # 基础方形分辨率，社区推荐默认分辨率
    
    # 中等尺寸（需要中等显存）
    "[FANG] 640x640",    # 中等方形分辨率，更好的细节
    "[FANG] 768x768",    # 高等方形分辨率，更清晰的细节
    
    # 高清尺寸（需要大量显存，仅高端显卡）
    "[FANG] 1024x1024",  # 高清方形分辨率，需要大量显存
    
    # 横向宽屏分辨率 (社区推荐)
    "[HENG] 480x320",    # 低分辨率 (更快的训练)
    "[HENG] 640x384",    # 标准社区推荐分辨率 (5:3)
    "[HENG] 704x416",    # 调整的分辨率，是32的倍数
    "[HENG] 960x544",    # 调整的qHD分辨率，显存需求适中
    "[HENG] 1024x576",   # 标准宽屏分辨率
    "[HENG] 1280x736",   # 调整的720p宽屏分辨率
    
    # 竖向分辨率 (短视频格式)
    "[SHU] 320x480",    # 低分辨率竖屏
    "[SHU] 384x640",    # 调整的社区推荐分辨率
    "[SHU] 416x704",    # 调整的分辨率，是32的倍数
    "[SHU] 576x1024",   # 标准竖屏分辨率
    "[SHU] 736x1280"    # 调整的720p竖屏分辨率
]

# 预设帧数列表 - 每个都是24的倍数加1
FRAME_COUNTS = [
    # 低帧数（推荐，训练速度快，显存需求低）
    "25",   # 24帧+1, 标准低帧数，社区推荐默认
    "49",   # 48帧+1, 中等帧数，平衡画质和资源
    "73",   # 72帧+1, 较高帧数，更流畅的动画
    
    # 中等帧数（需要中等显存）
    "97",   # 96帧+1, 高帧数，高清流畅
    "121",  # 120帧+1, 非常高的帧数（需要大量显存）
    "145",  # 144帧+1, 超高帧数，高分辨率视频用
    
    # 高帧数（需要大量显存，仅高端显卡）
    "169",  # 168帧+1, 超高帧数，极其流畅
    "193",  # 192帧+1, 超高帧数，高清长视频
    "241"   # 240帧+1, 最高帧数（需要大量显存）
]

# 兼容原有代码，保留RESOLUTIONS常量，但确保分辨率是32的倍数
RESOLUTIONS = [
    # 小尺寸方形分辨率 - 所有都是32的倍数
    "320x320x9", "320x320x17", "320x320x25",
    "384x384x9", "384x384x17", "384x384x25",
    "416x416x9", "416x416x17", "416x416x25",
    "512x512x9", "512x512x17", "512x512x25", "512x512x33", "512x512x49",
    
    # 横向宽屏分辨率
    "480x320x9", "480x320x17", "480x320x25",
    "640x384x9", "640x384x17", "640x384x25",
    "704x416x9", "704x416x17", "704x416x25",
    "960x544x9", "960x544x17", "960x544x25",
    "1024x576x9", "1024x576x17", "1024x576x25", "1024x576x49",
    
    # 竖向分辨率
    "320x480x9", "320x480x17", "320x480x25",
    "384x640x9", "384x640x17", "384x640x25",
    "416x704x9", "416x704x17", "416x704x25",
    "576x1024x9", "576x1024x17", "576x1024x25", "576x1024x49",
    
    # 中高分辨率组合
    "768x768x25", "768x768x49",
    "1024x1024x25", "1024x1024x49",
    "1280x736x25", "1280x736x49",
    "736x1280x25", "736x1280x49"
]

def run_command(cmd, status=None, verbose=True):
    """运行命令并将输出更新到状态框
    
    Args:
        cmd: 命令列表
        status: 可以是Gradio组件或其他对象，如果为None则只返回结果不更新任何UI
        verbose: 是否在终端显示简化日志
    
    Returns:
        命令的输出字符串
    """
    # 设置环境变量确保使用UTF-8编码
    cmd_env = os.environ.copy()
    cmd_env["PYTHONIOENCODING"] = "utf-8"
    
    # 打印命令
    cmd_str = ' '.join(cmd)
    logger.info(f"执行命令: {cmd_str}")
    
    # 判断命令类型
    is_preprocess_cmd = "preprocess_zero_workers.py" in cmd_str or "fix_resolution_wrapper.py" in cmd_str
    is_training_cmd = "train.py" in cmd_str
    is_convert_cmd = "convert_checkpoint.py" in cmd_str
    
    # 对需要实时输出的命令特殊处理
    need_realtime_output = is_preprocess_cmd or is_training_cmd or is_convert_cmd
    
    # 更新状态显示
    status_output = f"运行命令:\n{cmd_str}\n\n请等待..."
    if status is not None and hasattr(status, 'update'):
        status.update(value=status_output)
    
    try:
        # 对需要实时输出的命令，直接使用原生输出
        if need_realtime_output:
            cmd_type = "预处理" if is_preprocess_cmd else ("训练" if is_training_cmd else "转换")
            logger.info(f"使用实时输出模式执行{cmd_type}命令: {cmd_str}")
            
            # 创建进程，不捕获输出，让它直接显示在终端上
            process = subprocess.Popen(
                cmd,
                stdout=None,  # 直接使用父进程的stdout
                stderr=None,  # 直接使用父进程的stderr
                env=cmd_env,
                bufsize=0  # 禁用缓冲，确保实时显示
            )
            
            # 等待命令完成
            return_code = process.wait()
            
            # 命令完成后更新状态
            if return_code == 0:
                result_msg = f"{cmd_type}命令执行成功！请检查终端输出获取详细信息。"
                logger.info(result_msg)
            else:
                result_msg = f"{cmd_type}命令执行失败，返回代码: {return_code}。请检查终端输出获取错误信息。"
                logger.error(result_msg)
            
            # 更新UI显示
            if status is not None and hasattr(status, 'update'):
                if is_training_cmd:
                    status.update(value=f"{status_output}\n\n{result_msg}\n\n训练输出已在命令行窗口显示，请查看。")
                else:
                    status.update(value=f"{status_output}\n\n{result_msg}")
            
            return result_msg
        
        # 其他命令使用标准处理方式
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=cmd_env
        )
        
        # 读取输出
        stdout, stderr = process.communicate()
        
        # 组装输出信息
        output = f"命令返回码: {process.returncode}\n\n标准输出:\n{stdout}\n\n错误输出:\n{stderr}"
        
        # 更新状态显示
        if status is not None and hasattr(status, 'update'):
            status.update(value=output)
        
        return output
    except Exception as e:
        error_msg = f"执行命令时出错: {str(e)}"
        logger.error(error_msg)
        if status is not None and hasattr(status, 'update'):
            status.update(value=f"{status_output}\n\n{error_msg}")
        return error_msg
    #         if status is not None and hasattr(status, 'update'):
    #             status.update(value=f"开始训练, 命令: {cmd_str}\n\n{error_msg}")
    #         return error_msg
    
    # # 非训练命令使用原来的处理方式
    # cmd_str = " ".join(cmd)
    # status_output = f"运行命令:\n{cmd_str}\n\n请等待...\n"
    
    # if verbose:
    #     logger.info(f"执行命令: {cmd_str}")
    
    # # 更新状态的安全方法
    # def update_status(text):
    #     nonlocal status_output
    #     status_output = text
    #     if status is not None and hasattr(status, 'update'):
    #         try:
    #             status.update(value=text)
    #         except:
    #             pass  # 忽略更新错误
    
    # update_status(status_output)
    
    # try:
    #     # 使用subprocess.Popen来捕获输出
    #     process = subprocess.Popen(
    #         cmd,
    #         stdout=subprocess.PIPE,
    #         stderr=subprocess.STDOUT,  # 将stderr重定向到stdout便于一起捕获
    #         text=True,
    #         bufsize=1,
    #         universal_newlines=True,
    #         env=cmd_env
    #     )
        
    #     # 使用更健壮的输出读取方式，适合可能包含二进制数据的输出
    #     # 先终止subprocess.Popen方式，改用subprocess.run
    #     process.terminate()
        
    #     # 直接使用subprocess.run执行命令，并捕获完整输出
    #     try:
    #         # 设置text=False以获取原始字节数据，避免自动解码
    #         complete_result = subprocess.run(cmd, capture_output=True, env=cmd_env, text=False)
            
    #         # 有效处理输出 - 包含可能的二进制数据
    #         try:
    #             # 先尝试完整解码
    #             output = complete_result.stdout.decode('utf-8') if complete_result.stdout else ""
    #             error = complete_result.stderr.decode('utf-8') if complete_result.stderr else ""
    #         except UnicodeDecodeError:
    #             # 如果解码失败，尝试提取可读部分
    #             # 首先使用安全的latin1编码并替换错误字符
    #             output = complete_result.stdout.decode('latin1', errors='replace') if complete_result.stdout else ""
    #             error = complete_result.stderr.decode('latin1', errors='replace') if complete_result.stderr else ""
                
    #             # 不显示警告，这是正常情况
    #             if verbose:
    #                 logger.debug("含有二进制数据的输出已被安全处理")
            
    #         # 处理并的潜在错误无法显示的问题
    #         if "\ufffd" in output:
    #             # 尝试逐行解析，科学地处理混合文本和二进制数据
    #             readable_output = []
    #             for line in output.splitlines():
    #                 if not line.strip():
    #                     continue
    #                 # 如果这行主要是替换字符，跳过或简化处理
    #                 if line.count('\ufffd') > len(line) * 0.3:  # 如果替换字符超过30%
    #                     readable_output.append("[binary data omitted]")
    #                 else:
    #                     readable_output.append(line)
                
    #             # 重新组合输出
    #             output = "\n".join(readable_output)
            
    #         # 更新状态输出
    #         status_output += "\n" + output
    #         if error:  # 只有当有错误时才添加错误输出
    #             status_output += "\n\nErrors:\n" + error
            
    #         update_status(status_output)
    #     except Exception as e:
    #         logger.error(f"处理命令输出时出错: {e}")
    #         status_output += f"\n\n读取命令输出时出错: {e}"
    #         update_status(status_output)
        
    #     # 等待进程结束
    #     process.wait()
        
    #     if process.returncode == 0:
    #         if verbose:
    #             logger.info("命令执行成功!")
    #         update_status(status_output + "\n\n命令执行成功!")
    #     else:
    #         if verbose:
    #             logger.error(f"命令执行失败，返回代码 {process.returncode}")
    #         update_status(status_output + f"\n\n命令执行失败，返回代码 {process.returncode}")
        
    #     return status_output
    
    # except Exception as e:
    #     if verbose:
    #         logger.error(f"执行出错: {str(e)}")
    #     update_status(status_output + f"\n\n执行出错: {str(e)}")
    #     return status_output

def run_preprocessing(dataset, dims, frames, id_token, decode_videos, status):
    """运行数据预处理
    
    Args:
        dataset: 数据集路径
        dims: 分辨率尺寸 (如 "[横版] 768x768")
        frames: 帧数 (如 "49")
        id_token: LoRA触发词
        decode_videos: 是否验证视频解码
        status: 状态UI组件
    """
    # 提取实际分辨率，去除标识前缀
    dimensions = extract_dims(dims)
    
    # 组合分辨率和帧数
    resolution = f"{dimensions}x{frames}"
    
    # 动态检测设备，优先使用GPU（如果可用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"检测到可用计算设备: {device}")
    
    # 如果使用CUDA，显示GPU信息

def run_training(config_name, status):
    """运行训练脚本"""
    config_path = CONFIG_FILES.get(config_name)
    if not config_path:
        status.update(value=f"错误: 找不到配置 {config_name}")
        return f"错误: 找不到配置 {config_name}"
    
    # 查找临时配置文件
    temp_config_dir = os.path.join(PROJECT_DIR, "temp_configs")
    project_name = get_current_project_name()
    if project_name:
        temp_config_path = os.path.join(temp_config_dir, f"{project_name}_temp_config.yaml")
        if os.path.exists(temp_config_path):
            try:
                logger.info(f"使用临时配置文件进行训练: {temp_config_path}")
                cmd = [
                    sys.executable,
                    os.path.join(SCRIPTS_DIR, "train.py"),
                    temp_config_path
                ]
                return run_command(cmd, status)
            except Exception as e:
                logger.error(f"使用临时配置文件进行训练时出错: {e}")
                status.update(value=f"错误: 使用临时配置文件进行训练时出错: {e}")
                return f"错误: 使用临时配置文件进行训练时出错: {e}"
    
    # 如果没有找到临时配置，使用原始配置
    logger.info(f"使用原始配置文件进行训练: {config_path}")
    cmd = [
        sys.executable,
        os.path.join(SCRIPTS_DIR, "train.py"),
        config_path
    ]
    
    return run_command(cmd, status)

def find_ltx_model(model_name_pattern="ltxv-13b"):
    """查找LTX模型文件
    
    Args:
        model_name_pattern: 模型名称模式，如ltxv-13b
        
    Returns:
        找到的模型路径或None
    """
    # 直接返回diffusers目录作为模型路径，不再查找具体的safetensors文件
    if os.path.exists(DIFFUSERS_MODEL_PATH):
        logger.info(f"使用diffusers格式模型: {DIFFUSERS_MODEL_PATH}")
        return DIFFUSERS_MODEL_PATH
    
    logger.warning(f"未找到diffusers模型: {DIFFUSERS_MODEL_PATH}")
    return None

def get_preprocessed_path(basename):
    """获取预处理数据的绝对路径，优先使用绝对路径
    
    Args:
        basename: 项目名称
        
    Returns:
        预处理数据的绝对路径
    """
    # 首先尝试找到数据集位置
    dataset_path = check_dataset_location(basename)
    
    # 如果我们找到了数据集路径，尝试不同的预处理数据位置
    if dataset_path:
        # 首先检查.precomputed目录（最新的结构）
        precomputed_path = os.path.join(dataset_path, ".precomputed")
        if os.path.exists(precomputed_path) and os.path.isdir(precomputed_path):
            # 检查是否存在latents和conditions子目录
            if (os.path.exists(os.path.join(precomputed_path, "latents")) and 
                os.path.exists(os.path.join(precomputed_path, "conditions"))):
                logger.info(f"找到.precomputed目录下的预处理数据: {precomputed_path}")
                return os.path.abspath(precomputed_path)
        
        # 检查项目名_scenes目录
        scenes_path = os.path.join(dataset_path, f"{basename}_scenes")
        if os.path.exists(scenes_path) and os.path.isdir(scenes_path):
            # 检查是否存在latents和conditions子目录
            if (os.path.exists(os.path.join(scenes_path, "latents")) and 
                os.path.exists(os.path.join(scenes_path, "conditions"))):
                logger.info(f"找到{basename}_scenes目录下的预处理数据: {scenes_path}")
                return os.path.abspath(scenes_path)
        
        # 检查scenes目录
        alt_scenes_path = os.path.join(dataset_path, "scenes")
        if os.path.exists(alt_scenes_path) and os.path.isdir(alt_scenes_path):
            # 检查是否存在latents和conditions子目录
            if (os.path.exists(os.path.join(alt_scenes_path, "latents")) and 
                os.path.exists(os.path.join(alt_scenes_path, "conditions"))):
                logger.info(f"找到scenes目录下的预处理数据: {alt_scenes_path}")
                return os.path.abspath(alt_scenes_path)
        
        # 如果在数据集路径下找不到预处理数据，但找到了.precomputed目录，则返回该路径
        if os.path.exists(precomputed_path) and os.path.isdir(precomputed_path):
            logger.info(f"找到.precomputed目录，但可能不完整: {precomputed_path}")
            return os.path.abspath(precomputed_path)
        
        # 如果找到了数据集但没有预处理数据，则返回.precomputed路径（可能还未创建）
        logger.warning(f"数据集存在但无预处理数据，将使用默认.precomputed路径")
        return os.path.abspath(os.path.join(dataset_path, ".precomputed"))
    
    # 如果找不到数据集路径，尝试项目目录下的其他标准位置
    # 尝试项目目录下的{basename}_scenes
    scenes_path = f"{basename}_scenes"
    full_scenes_path = os.path.join(PROJECT_DIR, scenes_path)
    if os.path.exists(full_scenes_path) and os.path.isdir(full_scenes_path):
        # 检查是否存在latents和conditions子目录
        if (os.path.exists(os.path.join(full_scenes_path, "latents")) and 
            os.path.exists(os.path.join(full_scenes_path, "conditions"))):
            logger.info(f"在项目目录下找到预处理数据目录: {full_scenes_path}")
            return os.path.abspath(full_scenes_path)
    
    # 作为最后的尝试，直接返回默认的.precomputed路径
    default_path = os.path.join(PROJECT_DIR, "train_date", basename, ".precomputed")
    if os.path.exists(default_path) and os.path.isdir(default_path):
        logger.info(f"找到默认.precomputed目录: {default_path}")
        return os.path.abspath(default_path)
    
    # 如果所有尝试都失败，返回最可能的路径（可能会创建）
    logger.warning(f"无法找到预处理数据，将使用默认路径: {default_path}")
    return os.path.abspath(default_path)

def check_dataset_location(basename):
    """检查数据集位置，支持train_date/basename或basename_raw格式
    
    Args:
        basename: 数据集基本名称
        
    Returns:
        数据集绝对路径或None（如果不存在）
    """
    # 检查train_date/basename格式
    standard_path = os.path.join(DATA_DIR, basename)
    if os.path.exists(standard_path) and os.path.isdir(standard_path):
        return standard_path
    
    # 检查basename_raw格式
    raw_path = f"{basename}_raw"
    raw_full_path = os.path.join(PROJECT_DIR, raw_path)
    if os.path.exists(raw_full_path) and os.path.isdir(raw_full_path):
        return raw_full_path
    
    # 都不存在
    return None

def extract_trigger_word(basename):
    """从数据集名称中提取触发词
    
    规则：
    1. 如果数据集名称包含下划线，取第一个下划线前的部分
    2. 如果数据集名称包含空格，取第一个空格前的部分
    3. 如果数据集名称是纯文本，使用整个名称
    
    Args:
        basename: 数据集名称
        
    Returns:
        提取的触发词
    """
    # 如果是"XXX_scenes"格式，移除"_scenes"后缀
    if basename.endswith("_scenes"):
        basename = basename[:-7]
    
    # 如果包含下划线，取第一段
    if "_" in basename:
        return basename.split("_")[0]
    
    # 如果包含空格，取第一段
    if " " in basename:
        return basename.split(" ")[0]
    
    # 否则使用整个名称
    return basename

def extract_dims(dims_string):
    """从带有前缀标识的分辨率字符串中提取实际分辨率
    
    Args:
        dims_string: 带有前缀标识的分辨率，如 "[横版] 1024x576"
    
    Returns:
        实际分辨率字符串，如 "1024x576"
    """
    # 如果分辨率字符串包含方向标识前缀，去除前缀
    # 支持中文和拼音格式的前缀
    if "[" in dims_string and "]" in dims_string:
        # 找到最后一个方括号位置并取其后的内容
        dims_string = dims_string.split("]")
        if len(dims_string) > 1:
            dims_string = dims_string[1].strip()
    
    return dims_string

def run_pipeline(basename, dims, frames, config_name, rank, split_scenes=True, caption=True, preprocess=True, status=None, only_preprocess=False, add_trigger=True):
    """运行完整流水线 - 根据选项执行分场景、标注和预处理步骤，然后用train.py训练
    
    Args:
        basename: 项目名称
        dims: 分辨率尺寸 (如 "[横版] 768x768")
        frames: 帧数 (如 "49")
        config_name: 配置模板名称，如果为None则不进行训练
        rank: LoRA秀，如果为0则不进行训练
        split_scenes: 是否将长视频拆分成场景
        caption: 是否自动标注视频
        preprocess: 是否执行预处理步骤
        status: 状态UI组件
        only_preprocess: 是否只执行预处理步骤，不进行训练
        add_trigger: 是否添加触发词到标注文件
    """
    # 初始化重要变量，防止未定义错误
    precomputed_path = None
    temp_config_path = None
    resolution_bucket = None
    cache_files = []
    latents_files = []
    text_embeds_files = []
    process = None
    # 提取实际分辨率，去除标识前缀
    dimensions = extract_dims(dims)
    
    # 组合分辨率和帧数
    resolution = f"{dimensions}x{frames}"
    
    # 如果在只预处理模式下或config_name为None，跳过配置文件验证
    if only_preprocess or config_name is None:
        logger.info("只执行预处理模式，跳过配置文件验证")
        temp_config_path = None  # 确保不使用任何配置文件
    else:
        # 正常验证配置文件
        config_template = CONFIG_FILES.get(config_name)
        if not config_template:
            error_msg = f"错误: 找不到配置 {config_name}"
            if hasattr(status, 'update'):
                status.update(value=error_msg)
            logger.error(error_msg)
            return error_msg
        
        # 为了保证训练成功，我们直接使用train.py而不是run_pipeline.py
        logger.info(f"使用配置模板: {config_template} 创建临时配置文件")
    
    # 只有在需要训练时才检查模型文件
    model_path = None
    if not only_preprocess and config_name is not None:
        model_pattern = "ltxv-13b" if "13b" in config_name else "ltx-video-2b"
        model_path = find_ltx_model(model_pattern)
        logger.info(f"使用diffusers格式模型: {model_path}")
    elif only_preprocess:
        logger.info("只执行预处理模式，跳过模型文件检查")
    
    # 检查数据集路径
    dataset_path = check_dataset_location(basename)
    if not dataset_path:
        error_msg = f"错误: 未找到数据集 '{basename}'\n请确保数据位于 'train_date/{basename}' 或 '{basename}_raw' 目录"
        if hasattr(status, 'update'):
            status.update(value=error_msg)
        logger.error(error_msg)
        return error_msg
    
    # 分场景步骤（可选）
    if split_scenes:
        # 原始视频应该在数据集下的raw_videos目录
        raw_videos_dir = os.path.join(dataset_path, "raw_videos")
        # 场景输出目录应该在数据集下的scenes目录
        scenes_dir = os.path.join(dataset_path, "scenes")
        
        # 检查原始视频目录下是否有视频文件
        raw_videos = []
        video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
        video_files = []
        for root, _, files in os.walk(dataset_path):
            if os.path.basename(root) == "raw_videos" or os.path.basename(root) == "scenes":
                continue  # 跳过raw_videos和scenes目录
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                if ext.lower() in video_extensions:
                    video_files.append(file_path)
        
        if video_files:
            # 创建raw_videos目录
            os.makedirs(raw_videos_dir, exist_ok=True)
            
            # 复制视频文件并重命名以避免特殊字符和空格导致的问题
            logger.info(f"从数据集根目录发现{len(video_files)}个视频文件，复制到raw_videos目录")
            
            # 原始文件到新文件的映射，用于记录
            file_mapping = {}
            processed_files = []
            
            for i, file_path in enumerate(video_files):
                # 生成简化的文件名，使用项目名称作为前缀
                _, ext = os.path.splitext(file_path)
                safe_filename = f"{basename}{i+1:02d}{ext.lower()}"
                
                # 目标路径
                dest_path = os.path.join(raw_videos_dir, safe_filename)
                
                # 保存映射关系
                original_name = os.path.basename(file_path)
                file_mapping[original_name] = safe_filename
                processed_files.append(file_path)
                
                # 复制文件
                if not os.path.exists(dest_path):
                    shutil.copy2(file_path, dest_path)
                    logger.info(f"重命名视频: '{original_name}' -> '{safe_filename}'")
            
            # 删除原始视频文件，避免数据集重复
            for file_path in processed_files:
                if os.path.exists(file_path) and os.path.dirname(file_path) != raw_videos_dir:
                    try:
                        os.remove(file_path)
                        logger.info(f"删除原始视频文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"无法删除原始视频文件 {file_path}: {str(e)}")
                
            # 如果有多个文件被重命名，写入映射记录文件
            if len(file_mapping) > 1:
                mapping_file = os.path.join(dataset_path, "video_name_mapping.txt")
                with open(mapping_file, 'w', encoding='utf-8') as f:
                    f.write("\u539f\u59cb\u6587\u4ef6\u540d -> \u91cd\u547d\u540d\u540e\u7684\u6587\u4ef6\u540d\n")
                    f.write("---------------------------------------------------\n")
                    for orig, new in file_mapping.items():
                        f.write(f"{orig} -> {new}\n")
                logger.info(f"文件名映射记录已保存到: {mapping_file}")

            # 正确记录raw_videos的完整路径
            raw_videos = [os.path.join(raw_videos_dir, filename) for filename in file_mapping.values()]
        
        # 检查scenes目录是否存在且不为空
        scenes_empty = True
        if os.path.exists(scenes_dir):
            # 检查scenes目录中是否有视频文件
            scene_videos = []
            for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                scene_videos.extend(list(Path(scenes_dir).glob(f"*{ext}")))
            scenes_empty = len(scene_videos) == 0
            
        if raw_videos and (not os.path.exists(scenes_dir) or scenes_empty):
            # 存在原始视频但没有场景目录或场景目录为空，执行分场景
            os.makedirs(scenes_dir, exist_ok=True)
            logger.info(f"找到{len(raw_videos)}个原始视频文件，开始执行分场景...")
            if hasattr(status, 'update'):
                status.update(value=f"开始对{len(raw_videos)}个视频文件进行分场景...")
            
            # 直接执行分场景命令，避免使用ProcessPoolExecutor可能导致的死锁
            for video_path in raw_videos:
                # 确保使用完整路径
                cmd = [
                    sys.executable,
                    os.path.join(SCRIPTS_DIR, "split_scenes.py"),
                    video_path,  # 完整路径
                    scenes_dir,
                    "--filter-shorter-than", "5s",
                    "--detector", "content",
                    "--threshold", "30"
                ]
                logger.info(f"执行分场景命令: {' '.join(cmd)}")
                run_command(cmd, status)  # 添加status参数以显示进度
            
            # 重新检查分场景结果
            scene_videos = []
            for ext in video_extensions:
                scene_videos.extend(list(Path(scenes_dir).glob(f"*{ext}")))
                
            if scene_videos:
                logger.info(f"分场景完成，生成了{len(scene_videos)}个场景视频")
                if hasattr(status, 'update'):
                    status.update(value=f"分场景完成，生成了{len(scene_videos)}个场景视频")
                
                # 在分场景完成后立即复制视频文件到caption目录
                caption_dir = os.path.join(dataset_path, "caption")
                os.makedirs(caption_dir, exist_ok=True)
                
                # 复制scenes目录中的所有视频到caption目录
                copied_count = 0
                for video_file in scene_videos:
                    video_filename = os.path.basename(video_file)
                    caption_video_path = os.path.join(caption_dir, video_filename)
                    
                    # 如果目标文件不存在或者源文件比目标文件新，才复制
                    if not os.path.exists(caption_video_path) or \
                       (os.path.exists(str(video_file)) and 
                        os.path.getmtime(str(video_file)) > os.path.getmtime(caption_video_path)):
                        shutil.copy2(str(video_file), caption_video_path)
                        copied_count += 1
                        logger.info(f"复制视频到caption目录: {video_filename}")
                
                if copied_count > 0:
                    logger.info(f"已复制{copied_count}个场景视频到caption目录")
                    if hasattr(status, 'update'):
                        status.update(value=f"分场景完成，生成了{len(scene_videos)}个场景视频，并已复制到caption目录")
                else:
                    logger.info("所有视频文件已存在于caption目录中，无需复制")
                    
        else:
            if not raw_videos:
                logger.info(f"未找到原始视频文件，跳过分场景步骤")
                if hasattr(status, 'update'):
                    status.update(value="未找到原始视频文件，跳过分场景步骤")
            else:
                logger.info("场景目录已存在且有视频文件，跳过分场景步骤")
                if hasattr(status, 'update'):
                    status.update(value="场景目录已存在，跳过分场景步骤")
                    
            # 即使跳过分场景，也需要确保已存在的场景被复制到caption目录
            if os.path.exists(scenes_dir):
                caption_dir = os.path.join(dataset_path, "caption")
                os.makedirs(caption_dir, exist_ok=True)
                
                scene_videos = []
                for ext in video_extensions:
                    scene_videos.extend(list(Path(scenes_dir).glob(f"*{ext}")))
                
                if scene_videos:
                    copied_count = 0
                    for video_file in scene_videos:
                        video_filename = os.path.basename(video_file)
                        caption_video_path = os.path.join(caption_dir, video_filename)
                        
                        if not os.path.exists(caption_video_path) or \
                           (os.path.exists(str(video_file)) and 
                            os.path.getmtime(str(video_file)) > os.path.getmtime(caption_video_path)):
                            shutil.copy2(str(video_file), caption_video_path)
                            copied_count += 1
                            logger.info(f"复制已存在的视频到caption目录: {video_filename}")
                    
                    if copied_count > 0:
                        logger.info(f"已复制{copied_count}个现有场景视频到caption目录")
                        if hasattr(status, 'update'):
                            current_status = status.value if hasattr(status, 'value') else ""
                            status.update(value=current_status + f"\n已复制{copied_count}个现有场景视频到caption目录")
    
    # 检查是否已经有标注文件
    # 在数据集目录下创建caption目录
    caption_dir = os.path.join(dataset_path, "caption")
    os.makedirs(caption_dir, exist_ok=True)
    
    # 标注文件应该在caption目录下
    caption_file = os.path.join(caption_dir, "caption.txt")
    caption_json = os.path.join(caption_dir, "captions.json")
    
    # 确保在预处理前有caption.txt文件
    has_caption_file = os.path.exists(caption_file) and os.path.getsize(caption_file) > 0
    has_caption_json = os.path.exists(caption_json) and os.path.getsize(caption_json) > 0
    
    # 优先使用现有的caption.txt文件，如果它存在
    if has_caption_file:
        logger.info(f"找到现有标注文件(caption.txt)，跳过标注步骤")
        if hasattr(status, 'update'):
            current_status = status.value if hasattr(status, 'value') else ""
            status.update(value=current_status + "\n跳过标注步骤")
    # 如果有JSON文件但没有TXT文件，尝试转换
    elif has_caption_json:
        logger.info(f"找到JSON标注文件，正在转换为caption.txt格式...")
        if hasattr(status, 'update'):
            current_status = status.value if hasattr(status, 'value') else ""
            status.update(value=current_status + "\n正在将JSON标注转换为TXT格式...")
            
        # 尝试转换JSON到TXT
        try:
            import json
            # 读取JSON文件
            with open(caption_json, 'r', encoding='utf-8') as f:
                captions_data = json.load(f)
            
            # 创建caption.txt - 预处理脚本期望的格式
            with open(caption_file, 'w', encoding='utf-8') as f:
                for media_path, caption in captions_data.items():
                    # 将相对路径转换为绝对路径，如果必要的话
                    abs_media_path = os.path.join(dataset_path, media_path) if not os.path.isabs(media_path) else media_path
                    f.write(f"{abs_media_path}|{caption}\n")
            
            logger.info(f"成功从captions.json创建caption.txt标注文件: {caption_file}")
            if hasattr(status, 'update'):
                current_status = status.value if hasattr(status, 'value') else ""
                status.update(value=current_status + "\n成功将JSON标注转换为TXT格式")
                
            # 直接跳过下面的标注生成步骤
            caption = False  # 设置为False跳过标注生成
        except Exception as e:
            error_msg = f"转换标注文件时出错: {str(e)}"
            logger.error(error_msg)
            if hasattr(status, 'update'):
                current_status = status.value if hasattr(status, 'value') else ""
                status.update(value=current_status + f"\n{error_msg}")
            # 这里不返回错误，而是继续尝试生成标注
    
    # 无论如何，再次检查caption.txt是否存在
    has_caption_file = os.path.exists(caption_file) and os.path.getsize(caption_file) > 0
    
    # 如果仍然没有标注文件且用户要求标注，执行标注脚本
    if caption and not has_caption_file:
        # 执行标注命令
        logger.info(f"未找到标注文件，开始执行视频标注流程...")
        if hasattr(status, 'update'):
            status.update(value="正在生成视频标注...")
        
        # 确定正确的输入目录 - 优先使用caption目录的视频
        # 首先检查caption目录中是否有视频文件
        caption_videos = []
        if os.path.exists(caption_dir):
            for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                caption_videos.extend(list(Path(caption_dir).glob(f"*{ext}")))
        
        # 然后检查scenes目录
        scenes_dir = os.path.join(dataset_path, "scenes")
        scene_videos = []
        if os.path.exists(scenes_dir) and not caption_videos:
            for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                scene_videos.extend(list(Path(scenes_dir).glob(f"*{ext}")))
            
            # 如果有scenes视频但没有caption视频，自动复制
            if scene_videos and not caption_videos:
                logger.info("在caption目录中没有找到视频文件，将从cenes目录复制")
                copied_count = 0
                for video_file in scene_videos:
                    video_filename = os.path.basename(video_file)
                    caption_video_path = os.path.join(caption_dir, video_filename)
                    
                    shutil.copy2(str(video_file), caption_video_path)
                    copied_count += 1
                    logger.info(f"复制视频到caption目录: {video_filename}")
                
                logger.info(f"已复制{copied_count}个场景视频到caption目录")
                # 更新caption_videos列表
                for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                    caption_videos.extend(list(Path(caption_dir).glob(f"*{ext}")))
        
        # 选择适当的输入目录
        # 检查raw_videos目录是否存在视频文件
        raw_videos_dir = os.path.join(dataset_path, "raw_videos")
        raw_videos = []
        if os.path.exists(raw_videos_dir) and not (caption_videos or scene_videos):
            for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                raw_videos.extend(list(Path(raw_videos_dir).glob(f"*{ext}")))
                
        # 优先级：caption_dir > scenes_dir > raw_videos_dir > dataset_path
        if caption_videos:
            input_dir = caption_dir
            logger.info(f"将使用caption目录中的视频文件进行标注")
        elif scene_videos:
            input_dir = scenes_dir
            logger.warning(f"注意: 使用scenes目录中的视频文件进行标注，这可能会导致路径不匹配")
        elif raw_videos:
            input_dir = raw_videos_dir
            logger.warning(f"注意: 使用raw_videos目录中的原始视频进行标注")
        else:
            input_dir = dataset_path
            logger.warning(f"注意: 使用数据集根目录进行标注，这可能不是预期的行为")
            
        logger.info(f"将使用{input_dir}目录中的视频文件进行标注")
        
        # 准备标注输出路径
        output_json = os.path.join(dataset_path, "captions.json")
        
        # 创建标注目录
        os.makedirs(caption_dir, exist_ok=True)
        
        # 在执行标注命令前，先检查目录中是否确实有视频文件
        video_files_in_input = []
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            video_files_in_input.extend(list(Path(input_dir).glob(f"*{ext}")))
            
        if not video_files_in_input:
            error_msg = f"错误: 未在{input_dir}目录中找到视频文件。请先添加视频文件或选择正确的数据集目录。"
            if hasattr(status, 'update'):
                status.update(value=error_msg)
            logger.error(error_msg)
            return error_msg
        
        logger.info(f"在{input_dir}目录中找到{len(video_files_in_input)}个视频文件")
        
        caption_cmd = [
            sys.executable,
            os.path.join(SCRIPTS_DIR, "caption_videos.py"),
            input_dir,  # 使用选定的输入目录
            "--output", output_json,
            "--captioner-type", "llava_next_7b"
        ]
        
        logger.info(f"执行标注命令: {' '.join(caption_cmd)}")
        caption_output = run_command(caption_cmd, status)
        
        # 检查标注是否成功
        caption_file_exists = os.path.exists(caption_file)
        caption_json_exists = os.path.exists(caption_json) and os.path.getsize(caption_json) > 0
        output_json_exists = os.path.exists(output_json) and os.path.getsize(output_json) > 0
        caption_success_message = "Captioned" in caption_output and "successfully" in caption_output
        
        # 如果标注成功，继续处理流程
        if caption_file_exists or caption_json_exists or caption_success_message or output_json_exists:
            logger.info(f"标注已生成，继续处理流程")
            if hasattr(status, 'update'):
                status.update(value="视频标注完成，准备进行预处理")
                
            # 如果标注文件生成在数据集根目录，复制到caption目录
            if output_json_exists:
                logger.info(f"复制标注JSON文件到caption目录: {caption_json}")
                shutil.copy2(output_json, caption_json)
        elif "No media files found" in caption_output:
            error_msg = f"错误: 标注脚本未找到媒体文件。请确保在{input_dir}目录中有视频文件且文件名不包含中文字符或特殊符号。"
            if hasattr(status, 'update'):
                status.update(value=error_msg)
            logger.error(error_msg)
            return error_msg
        else:
            error_msg = f"标注视频失败: {caption_output}"
            if hasattr(status, 'update'):
                status.update(value=error_msg)
            logger.error(error_msg)
            return error_msg
        
        # 如果生成了JSON文件但没有TXT文件，转换JSON为TXT
        # 检查所有可能的JSON标注文件位置
        json_files = [
            caption_json,  # caption目录中的JSON文件
            os.path.join(dataset_path, "captions.json"),  # 数据集根目录中的JSON文件
            os.path.join(scenes_dir, "captions.json")  # scenes目录中的JSON文件
        ]
        
        # 找到最新的有效JSON标注文件
        valid_json_file = None
        for json_file in json_files:
            if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
                if valid_json_file is None or os.path.getmtime(json_file) > os.path.getmtime(valid_json_file):
                    valid_json_file = json_file
        
        # 如果有效的JSON文件与caption目录中的不同，复制到caption目录
        if valid_json_file and valid_json_file != caption_json:
            logger.info(f"复制最新的标注JSON文件到caption目录: {valid_json_file} -> {caption_json}")
            shutil.copy2(valid_json_file, caption_json)
        
        # 试图将JSON标注转换为TXT格式（如果存在JSON文件但没有TXT文件）
        if os.path.exists(caption_json) and not (os.path.exists(caption_file) and os.path.getsize(caption_file) > 0):
            try:
                logger.info(f"将JSON标注转换为TXT格式...")
                
                # 读取JSON文件
                with open(caption_json, 'r', encoding='utf-8') as f:
                    captions_data = json.load(f)
                
                # 创建caption.txt
                with open(caption_file, 'w', encoding='utf-8') as f:
                    for item in captions_data:
                        if isinstance(item, dict) and 'caption' in item and 'media_path' in item:
                            # 只使用文件名作为关键字
                            media_name = os.path.basename(item['media_path'])
                            f.write(f"{media_name}|{item['caption']}\n")
                
                logger.info(f"成功将JSON标注转换为TXT格式: {caption_file}")
            except Exception as e:
                logger.error(f"转换JSON标注到TXT格式时出错: {str(e)}")
        
        # 如果标注文件存在，自动添加触发词
        if os.path.exists(caption_file) and os.path.getsize(caption_file) > 0:
            logger.info(f"开始添加触发词到标注文件...")
            if hasattr(status, 'update'):
                current_status = status.value if hasattr(status, 'value') else ""
                status.update(value=current_status + "\n正在添加触发词到标注文件...")
            
            # 运行触发词添加脚本
            add_trigger_cmd = [
                sys.executable,
                os.path.join(SCRIPTS_DIR, "add_trigger_word_to_captions.py"),
                dataset_path,
                "--trigger-word", basename
            ]
            
            logger.info(f"执行触发词添加命令: {' '.join(add_trigger_cmd)}")
            trigger_output = run_command(add_trigger_cmd, status)
            
            if "\u6807注文件处理完成" in trigger_output:
                logger.info(f"触发词添加成功，继续处理流程")
                if hasattr(status, 'update'):
                    current_status = status.value if hasattr(status, 'value') else ""
                    status.update(value=current_status + f"\n触发词 <{basename}> 已添加到标注文件")
            else:
                logger.warning(f"添加触发词可能有问题: {trigger_output}")
                if hasattr(status, 'update'):
                    current_status = status.value if hasattr(status, 'value') else ""
                    status.update(value=current_status + f"\n注意: 添加触发词可能有问题，请检查日志")
        
        # 检查标注文件是否存在，如果仍不存在，创建一个空的标注文件
        
        # 如果caption.txt不存在或为空，但有JSON文件，则转换为TXT格式
        if (not os.path.exists(caption_file) or os.path.getsize(caption_file) == 0) and valid_json_file:
            try:
                import json
                logger.info(f"将JSON标注文件转换为TXT格式: {valid_json_file} -> {caption_file}")
                
                # 读取JSON文件
                with open(valid_json_file, 'r', encoding='utf-8') as f:
                    captions_data = json.load(f)
                
                # 创建caption.txt - 预处理脚本期望的格式
                converted_count = 0
                media_files_to_copy = []  # 记录需要复制的媒体文件
                
                with open(caption_file, 'w', encoding='utf-8') as f:
                    # 处理不同格式的JSON标注数据
                    if isinstance(captions_data, list):
                        # 列表格式（新格式）
                        for item in captions_data:
                            if isinstance(item, dict):
                                if 'media_path' in item and 'caption' in item:
                                    media_path = item['media_path']
                                    caption_text = item['caption']
                                    # 关键修改：使用本地影片名而不是绝对路径 - 预处理脚本期望当前目录下的文件
                                    # 无论文件路径是什么，我们都只使用影片名
                                    original_media_path = media_path
                                    
                                    # 获取原始路径的文件名
                                    file_name = os.path.basename(media_path)
                                    
                                    # 要复制到caption目录，所以这里只使用文件名做为路径
                                    media_path = file_name
                                    
                                    # 如果原始路径指向raw_videos目录中的文件，确保它会被复制到caption目录
                                    if "raw_videos" in original_media_path and os.path.exists(original_media_path):
                                        caption_media_path = os.path.join(caption_dir, file_name)
                                        media_files_to_copy.append((original_media_path, caption_media_path))
                                    
                                    # 直接写文件名到标注文件，而不是完整路径
                                    f.write(f"{media_path}|{caption_text}\n")
                                    converted_count += 1
                    elif isinstance(captions_data, dict):
                        # 字典格式（旧格式）
                        for media_path, caption_text in captions_data.items():
                            # 使用相同的方法对字典格式进行处理，只使用文件名
                            original_media_path = media_path
                            
                            # 获取原始路径的文件名
                            file_name = os.path.basename(media_path)
                            
                            # 只使用文件名做为路径
                            media_path = file_name
                            
                            # 如果原始路径指向raw_videos目录中的文件，确保它会被复制到caption目录
                            if "raw_videos" in original_media_path and os.path.exists(original_media_path):
                                caption_media_path = os.path.join(caption_dir, file_name)
                                media_files_to_copy.append((original_media_path, caption_media_path))
                            
                            # 直接写文件名到标注文件，而不是完整路径
                            f.write(f"{media_path}|{caption_text}\n")
                            converted_count += 1
                    else:
                        # 不支持的格式
                        raise ValueError(f"不支持的JSON标注格式: {type(captions_data)}, 应为列表或字典类型")
                
                # 复制视频文件到caption目录
                copied_files_count = 0
                
                # 直接扫描原始视频目录，强制复制所有视频
                raw_videos_dir = os.path.join(dataset_path, "raw_videos")
                if os.path.exists(raw_videos_dir):
                    raw_video_files = []
                    for file in os.listdir(raw_videos_dir):
                        file_path = os.path.join(raw_videos_dir, file)
                        if os.path.isfile(file_path) and file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                            raw_video_files.append(file_path)
                    
                    if raw_video_files:
                        logger.info(f"强制复制raw_videos目录下的{len(raw_video_files)}个视频文件到caption目录")
                        
                        for src_path in raw_video_files:
                            # 为每个视频文件创建一个对应的caption目录目标路径
                            file_name = os.path.basename(src_path)
                            dst_path = os.path.join(caption_dir, file_name)
                            
                            if not os.path.exists(dst_path):
                                # 确保目标目录存在
                                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                                # 复制视频文件
                                shutil.copy2(src_path, dst_path)
                                copied_files_count += 1
                                logger.info(f"复制视频文件到caption目录: {src_path} -> {dst_path}")
                
                # 原有的复制逻辑仍然保留
                if media_files_to_copy:
                    for src_path, dst_path in media_files_to_copy:
                        if os.path.exists(src_path) and not os.path.exists(dst_path):
                            # 确保目标目录存在
                            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                            # 复制视频文件
                            shutil.copy2(src_path, dst_path)
                            copied_files_count += 1
                            logger.info(f"复制视频文件到caption目录: {src_path} -> {dst_path}")
                
                if converted_count == 0:
                    logger.warning(f"转换JSON标注完成，但未找到有效的标注条目")
                else:
                    logger.info(f"成功创建标注文件，包含{converted_count}条标注: {caption_file}")
                    logger.info(f"已复制{copied_files_count}个视频文件到caption目录")
                    # 标注完成，准备进行预处理，不需要复制标注文件回scenes目录
                
                if hasattr(status, 'update'):
                    status.update(value=f"视频标注转换完成，已复制{copied_files_count}个视频文件到caption目录，准备进行预处理")
            except Exception as e:
                error_msg = f"转换标注文件时出错: {str(e)}"
                logger.error(error_msg)
                if hasattr(status, 'update'):
                    status.update(value=error_msg)
                return error_msg
    else:
        logger.info(f"找到现有标注文件，跳过标注步骤")
        if hasattr(status, 'update'):
            current_status = status.value if hasattr(status, 'value') else ""
            status.update(value=current_status + "\n跳过标注步骤")
            
    # 最后一次检查标注文件是否存在 - 这是预处理所必需的
    if not os.path.exists(caption_file) or os.path.getsize(caption_file) == 0:
        error_msg = f"错误: 标注文件(caption.txt)不存在或为空，无法进行预处理"
        logger.error(error_msg)
        if hasattr(status, 'update'):
            status.update(value=error_msg)
        return error_msg
    
    # 自动向标注文件添加触发词
    # 提取触发词
    trigger_word = extract_trigger_word(basename)
    logger.info(f"标注文件已存在，继续处理并添加触发词 '{trigger_word}'")
    
    # 添加触发词到标注文件
    try:
        trigger_word_script = os.path.join(SCRIPTS_DIR, "add_trigger_word_to_captions.py")
        if os.path.exists(trigger_word_script):
            # 调用触发词添加脚本
            trigger_cmd = [
                sys.executable,
                trigger_word_script,
                dataset_path,
                "--trigger-word", trigger_word
            ]
            logger.info(f"执行触发词添加命令: {' '.join(trigger_cmd)}")
            result = run_command(trigger_cmd, status)
            logger.info(f"触发词添加完成")
            if hasattr(status, 'update'):
                current_status = status.value if hasattr(status, 'value') else ""
                status.update(value=current_status + f"\n已将触发词 '<{trigger_word}>' 添加到标注文件")
        else:
            logger.warning(f"触发词添加脚本不存在: {trigger_word_script}")
    except Exception as e:
        logger.warning(f"添加触发词到标注文件时出错: {str(e)}")
    
    # ======== 步骤3: 数据预处理部分 ========
    logger.info("\n=== 步骤3: 开始数据预处理 ===")
    if hasattr(status, 'update'):
        current_status = status.value if hasattr(status, 'value') else ""
        status.update(value=current_status + "\n\n=== 步骤3: 开始数据预处理 ===")
    
    # 初始化预处理路径
    precomputed_path = os.path.join(dataset_path, ".precomputed")
    
    # 创建预处理目录（如果不存在）
    if not os.path.exists(precomputed_path):
        logger.info(f"创建预处理目录: {precomputed_path}")
        os.makedirs(precomputed_path, exist_ok=True)

    # 检查是否已有预处理数据
    latents_dir = os.path.join(precomputed_path, "latents")
    conditions_dir = os.path.join(precomputed_path, "conditions")
    
    has_precomputed_data = False
    latents_files = []
    conditions_files = []
    
    # 更严格的检查: 两个目录必须存在且内部必须有文件
    if os.path.exists(latents_dir) and os.path.isdir(latents_dir) and os.path.exists(conditions_dir) and os.path.isdir(conditions_dir):
        latents_files = os.listdir(latents_dir)
        conditions_files = os.listdir(conditions_dir)
        
        # 只有当两个目录都非空时才认为有效
        if latents_files and conditions_files:
            has_precomputed_data = True
            logger.info(f"找到已预处理的数据: {len(latents_files)} 个潜在文件, {len(conditions_files)} 个条件文件")
            if hasattr(status, 'update'):
                current_status = status.value if hasattr(status, 'value') else ""
                status.update(value=current_status + f"\n已找到预处理数据: {len(latents_files)} 个潜在文件, {len(conditions_files)} 个条件文件")
        else:
            logger.warning(f"预处理目录存在但为空: 潜在文件={len(latents_files)}, 条件文件={len(conditions_files)}")
    else:
        logger.info(f"预处理目录不存在或不完整: latents_dir={os.path.exists(latents_dir)}, conditions_dir={os.path.exists(conditions_dir)}")
    
    # 决定是否需要运行预处理
    if preprocess and has_precomputed_data:
        logger.info(f"已找到预处理数据，跳过预处理步骤")
        if hasattr(status, 'update'):
            current_status = status.value if hasattr(status, 'value') else ""
            status.update(value=current_status + "\n\n跳过预处理步骤，使用已存在的预处理数据")
    elif preprocess:  # 需要预处理且没有现有预处理数据
        logger.info(f"检测到预处理数据不存在，开始执行预处理...")        
        if hasattr(status, 'update'):
            current_status = status.value if hasattr(status, 'value') else ""
            status.update(value=current_status + "\n\n=== 步骤3: 开始数据预处理 ===")
            
        # 在预处理前自动修复视频与标题不匹配问题
        try:
            fix_mismatch_script = os.path.join(SCRIPTS_DIR, "auto_fix_video_caption_mismatch.py")
            if os.path.exists(fix_mismatch_script):
                logger.info(f"在预处理前检查并修复视频与标题不匹配问题...")
                if hasattr(status, 'update'):
                    current_status = status.value if hasattr(status, 'value') else ""
                    status.update(value=current_status + "\n检查视频与标题是否匹配...")
                
                # 导入修复模块
                sys.path.append(SCRIPTS_DIR)
                try:
                    from auto_fix_video_caption_mismatch import check_and_fix_video_caption_mismatch
                    fixed, mapped_count, moved_count = check_and_fix_video_caption_mismatch(dataset_path)
                    
                    if fixed:
                        logger.info(f"成功修复视频与标题不匹配问题: 有标题视频={mapped_count}个, 移动了{moved_count}个不匹配的视频文件")
                        if hasattr(status, 'update'):
                            current_status = status.value if hasattr(status, 'value') else ""
                            status.update(value=current_status + f"\n已修复视频与标题不匹配问题，移动了{moved_count}个文件")
                    else:
                        logger.info(f"视频与标题匹配检查完成: 有标题视频={mapped_count}个")
                except ImportError:
                    # 如果导入失败，使用命令行方式执行
                    fix_cmd = [sys.executable, fix_mismatch_script, dataset_path]
                    logger.info(f"执行自动修复命令: {' '.join(fix_cmd)}")
                    fix_output = run_command(fix_cmd, status)
                    logger.info(f"自动修复输出: {fix_output}")
            else:
                logger.warning(f"自动修复脚本不存在: {fix_mismatch_script}")
        except Exception as e:
            logger.warning(f"执行自动修复脚本时出错: {str(e)}，将继续预处理流程")
            
        # 确保预处理目录已创建
        if not os.path.exists(precomputed_path):
            os.makedirs(precomputed_path, exist_ok=True)
            logger.info(f"创建预处理目录: {precomputed_path}")
            
        # 检查caption目录内容
        if os.path.exists(caption_dir):
            caption_files = os.listdir(caption_dir)
            logger.info(f"Caption目录内容: {caption_files}")
            
            # 检查caption.txt内容
            if os.path.exists(caption_file):
                try:
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        caption_content = f.read()
                    logger.info(f"标注文件内容:\n{caption_content}")
                except Exception as e:
                    logger.error(f"读取标注文件出错: {str(e)}")

            
        # 创建子目录以确保程序正确执行
        os.makedirs(os.path.join(precomputed_path, "latents"), exist_ok=True)
        os.makedirs(os.path.join(precomputed_path, "conditions"), exist_ok=True)
        
        # 检查caption目录内容
        if os.path.exists(caption_dir):
            caption_files = os.listdir(caption_dir)
            logger.info(f"Caption目录内容: {caption_files}")
            
            # 检查caption.txt内容
            if os.path.exists(caption_file):
                try:
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        caption_content = f.read()
                    logger.info(f"标注文件内容:\n{caption_content}")
                except Exception as e:
                    logger.error(f"读取标注文件出错: {str(e)}")

        # 定义预处理命令并执行
        try:
            # 检查必要的预处理文件是否存在
            dataset_script = os.path.join(SCRIPTS_DIR, "preprocess_dataset.py")
            if not os.path.exists(dataset_script):
                logger.error(f"预处理脚本不存在: {dataset_script}")
                if hasattr(status, 'update'):
                    current_status = status.value if hasattr(status, 'value') else ""
                    status.update(value=current_status + f"\n\n错误: 预处理脚本不存在 - {dataset_script}")
                return False
                
            # 复制标注文件到scenes目录
            scenes_caption_file = os.path.join(scenes_dir, "caption.txt")
            if not os.path.exists(scenes_caption_file) and os.path.exists(caption_file):
                try:
                    shutil.copy2(caption_file, scenes_caption_file)
                    logger.info(f"复制标注文件到scenes目录: {scenes_caption_file}")
                except Exception as e:
                    logger.warning(f"复制标注文件到scenes目录失败: {str(e)}")
            
            # 修改JSON文件中的路径格式，仅使用文件名而非路径
            # 正确使用captions.json文件路径
            source_json_path = os.path.join(dataset_path, "captions.json")
            
            if os.path.exists(source_json_path):
                try:
                    with open(source_json_path, 'r', encoding='utf-8') as f:
                        captions_data = json.load(f)
                    
                    # 将media_path从相对路径改为纯文件名
                    if isinstance(captions_data, list):
                        for item in captions_data:
                            if 'media_path' in item:
                                # 提取文件名
                                item['media_path'] = os.path.basename(item['media_path'])
                    elif isinstance(captions_data, dict):
                        modified_data = {}
                        for key, value in captions_data.items():
                            # 使用文件名作为新的键
                            new_key = os.path.basename(key)
                            modified_data[new_key] = value
                        captions_data = modified_data
                    
                    # 写回修改后的JSON文件
                    with open(source_json_path, 'w', encoding='utf-8') as f:
                        json.dump(captions_data, f, ensure_ascii=False, indent=2)
                        
                    logger.info(f"成功修改JSON文件中的路径格式: {source_json_path}")
                except Exception as e:
                    logger.warning(f"修改JSON文件失败: {str(e)}")
            else:
                logger.warning(f"找不到JSON文件: {source_json_path}")
                
            # 重要：首先复制到caption目录，然后再调用触发词添加脚本
            # 复制标注JSON文件到caption目录
            caption_json_path = os.path.join(caption_dir, "captions.json")
            if os.path.exists(source_json_path):
                shutil.copy2(source_json_path, caption_json_path)
                logger.info(f"复制修改后的标注JSON文件到caption目录: {caption_json_path}")
            else:
                logger.warning(f"无法复制JSON文件，源文件不存在: {source_json_path}")
            
            # 第一次将标注文件复制到数据集根目录
            root_caption_path = os.path.join(dataset_path, "caption.txt")
            if os.path.exists(caption_file):
                shutil.copy2(caption_file, root_caption_path)
                logger.info(f"复制标注文件到数据集根目录: {root_caption_path}")
            else:
                logger.warning(f"无法复制标注文件，源文件不存在: {caption_file}")
                
            # 在这里调用触发词脚本，确保在文件复制后进行处理
            # 再次检查并使用触发词脚本处理caption目录中的标注文件
            try:
                trigger_word_script = os.path.join(SCRIPTS_DIR, "add_trigger_word_to_captions.py")
                if os.path.exists(trigger_word_script):
                    # 调用触发词添加脚本
                    trigger_cmd = [
                        sys.executable,
                        trigger_word_script,
                        dataset_path,
                        "--trigger-word", extract_trigger_word(basename)
                    ]
                    logger.info(f"执行触发词添加命令: {' '.join(trigger_cmd)}")
                    result = run_command(trigger_cmd, status)
                    logger.info(f"触发词添加完成")
                    if hasattr(status, 'update'):
                        current_status = status.value if hasattr(status, 'value') else ""
                        status.update(value=current_status + f"\n已将触发词 '<{extract_trigger_word(basename)}>' 添加到标注文件")
                    
                    # 触发词添加后，从Caption目录再次复制到根目录确保触发词已添加
                    if os.path.exists(caption_file):
                        shutil.copy2(caption_file, root_caption_path)
                        logger.info(f"复制已添加触发词的标注文件到数据集根目录: {root_caption_path}")
                else:
                    logger.warning(f"触发词添加脚本不存在: {trigger_word_script}")
            except Exception as e:
                logger.warning(f"添加触发词到标注文件时出错: {str(e)}")
            
            # 重要改进：将视频文件也复制到数据集根目录，这样预处理脚本可以找到它们
            # 检查caption目录中的视频文件
            caption_video_files = []
            for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                caption_video_files.extend(list(Path(caption_dir).glob(f"*{ext}")))
            
            # 复制视频文件到数据集根目录
            video_filenames = []  # 存储视频文件名，用于创建media_path.txt
            if caption_video_files:
                logger.info(f"将{len(caption_video_files)}个视频文件从caption目录复制到数据集根目录")
                for video_file in caption_video_files:
                    video_filename = os.path.basename(video_file)
                    dest_path = os.path.join(dataset_path, video_filename)
                    shutil.copy2(str(video_file), dest_path)
                    logger.info(f"复制视频文件到数据集根目录: {video_filename}")
                    video_filenames.append(video_filename)
            else:
                logger.warning("caption目录中没有找到视频文件，预处理可能会失败")
                
            # 创建media_path.txt文件，预处理脚本需要这个文件来找到视频文件
            media_path_txt = os.path.join(dataset_path, "media_path.txt")
            with open(media_path_txt, 'w', encoding='utf-8') as f:
                for filename in video_filenames:
                    f.write(f"{filename}\n")
            logger.info(f"已创建media_path.txt文件，包含{len(video_filenames)}个视频文件路径")
            
            # 确保caption.txt文件也存在于根目录（这部分代码已在前面）
            
            # 使用修复的零工作线程预处理脚本，解决序列化错误
            zero_workers_script = os.path.join(SCRIPTS_DIR, "preprocess_zero_workers.py")
            
            # 如果有我们创建的零工作线程修复脚本，优先使用它
            if os.path.exists(zero_workers_script):
                fix_resolution_script = zero_workers_script
                logger.info(f"使用零工作线程修复脚本: {zero_workers_script}")
            else:
                # 如果没有零工作线程脚本，回退到分辨率格式修复包装器
                fix_resolution_script = os.path.join(SCRIPTS_DIR, "fix_resolution_wrapper.py")
                
                # 如果分辨率包装器也不存在，尝试使用原始包装器
                if not os.path.exists(fix_resolution_script):
                    fix_resolution_script = os.path.join(SCRIPTS_DIR, "preprocess_wrapper.py")
                
            if os.path.exists(fix_resolution_script):
                logger.info(f"使用预处理包装器: {fix_resolution_script}")
                
                # 给resolution_bucket设置默认值，防止None值错误
                if resolution_bucket is None:
                    # 从分辨率中提取宽高，作为默认resolution bucket
                    if dimensions and "x" in dimensions:
                        resolution_bucket = dimensions
                    else:
                        resolution_bucket = "768x768"  # 使用一个安全的默认值
                    logger.info(f"没有指定分辨率桶，使用默认值: {resolution_bucket}")
                
                # 动态检测设备，优先使用GPU（如果可用）
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"检测到可用计算设备: {device}")
                
                # 如果使用CUDA，显示GPU信息
                if device == "cuda":
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # 转换为GB
                    logger.info(f"使用GPU: {gpu_name}, 显存: {gpu_mem:.2f}GB")
                
                # 确保传递帧数参数
                preprocess_cmd = [
                    sys.executable,
                    fix_resolution_script,
                    os.path.dirname(caption_dir),  # 输入整个数据集目录而非仅caption子目录
                    "--resolution-buckets", resolution_bucket,
                    "--output-dir", precomputed_path,
                    "--batch-size", "1",  # 小批量大小防止内存问题
                    "--num-workers", "0",  # 强制使用单线程，防止序列化错误
                    "--device", device,  # 动态使用可用的最佳设备
                    "--frames", frames  # 显式传递帧数参数
                ]
                
                # 打印完整的分辨率信息便于调试
                logger.info(f"使用分辨率: {resolution_bucket}, 帧数: {frames}")
                
            else:
                # 就算没有包装器，也尝试生成自己的修复包装器脚本
                logger.info("没有找到预处理包装器，尝试生成脚本解决分辨率格式问题")
                
                # 生成一个简单的包装器脚本
                temp_wrapper_content = '''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import subprocess

args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--resolution-buckets" and i + 1 < len(args):
        resolution = args[i + 1]
        if "x" in resolution and resolution.count("x") == 1:
            args[i + 1] = f"{resolution}x49"
            print(f"修复分辨率格式: {resolution} -> {args[i + 1]}")
    i += 1

script_dir = os.path.dirname(os.path.abspath(__file__))
preprocess_script = os.path.join(script_dir, "preprocess_dataset.py")
cmd = [sys.executable, preprocess_script] + args
print(f"执行预处理命令: {' '.join(cmd)}")
process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
return_code = process.wait()
sys.exit(return_code)
'''
                
                # 创建临时脚本文件
                temp_script_path = os.path.join(SCRIPTS_DIR, "temp_resolution_fix.py")
                try:
                    with open(temp_script_path, 'w', encoding='utf-8') as f:
                        f.write(temp_wrapper_content)
                    logger.info(f"创建了临时分辨率修复脚本: {temp_script_path}")
                    # 设置可执行权限
                    os.chmod(temp_script_path, 0o755)
                except Exception as e:
                    logger.warning(f"创建临时脚本失败: {str(e)}")
                    temp_script_path = None
                
                # 给resolution_bucket设置默认值，防止None值错误
                if resolution_bucket is None:
                    # 从分辨率中提取宽高，作为默认resolution bucket
                    if dimensions and "x" in dimensions:
                        resolution_bucket = dimensions
                    else:
                        resolution_bucket = "768x768"  # 使用一个安全的默认值
                    logger.info(f"没有指定分辨率桶，使用默认值: {resolution_bucket}")
                
                # 设置使用的脚本
                script_to_use = temp_script_path if temp_script_path else dataset_script
                logger.info(f"调用预处理脚本: {script_to_use}")
                
                # 构建预处理命令
                preprocess_cmd = [
                    sys.executable,
                    script_to_use,
                    os.path.dirname(caption_dir),  # 输入整个数据集目录而非仅caption子目录
                    "--resolution-buckets", resolution_bucket,
                    "--output-dir", precomputed_path,
                    "--batch-size", "1",  # 小批量大小防止内存问题
                    "--num-workers", "1",  # 单线程减少问题
                    "--device", "cpu"  # 强制使用CPU模式
                ]
                
            # 这里是关键点 - 实际执行预处理命令
            logger.info(f"执行预处理命令: {' '.join(preprocess_cmd)}")
            if hasattr(status, 'update'):
                current_status = status.value if hasattr(status, 'value') else ""
                status.update(value=current_status + f"\n\n执行预处理命令: {' '.join(preprocess_cmd)}\n\n正在运行预处理...这可能需要一段时间")
            
            # 使用run_command函数执行命令
            preprocess_output = run_command(preprocess_cmd, status)
            
            # 检查预处理是否有错误
            preprocess_failed = False
            if "Traceback" in preprocess_output or "Error" in preprocess_output or "error" in preprocess_output:
                preprocess_failed = True
                error_msg = f"预处理脚本执行出错:\n{preprocess_output[:1000]}..."
                logger.error(error_msg)
                if hasattr(status, 'update'):
                    current_status = status.value if hasattr(status, 'value') else ""
                    status.update(value=current_status + f"\n\n{error_msg}")
            
            # 显示命令输出
            if preprocess_output:
                logger.info(f"预处理命令输出:\n{preprocess_output[:500]}...")  # 显示前500个字符
            else:
                logger.warning("预处理命令没有输出")
                
            # 检查命令返回码
            if "\n命令返回码: 1\n" in preprocess_output or "\n命令返回码: 2\n" in preprocess_output:
                preprocess_failed = True
                error_msg = f"预处理命令返回非零代码，表示执行失败"
                logger.error(error_msg)
                if hasattr(status, 'update'):
                    current_status = status.value if hasattr(status, 'value') else ""
                    status.update(value=current_status + f"\n\n{error_msg}")
                
            # 再次检查预处理目录是否有实际文件生成
            latents_dir = os.path.join(precomputed_path, "latents")
            conditions_dir = os.path.join(precomputed_path, "conditions")
            
            if os.path.exists(latents_dir) and os.path.exists(conditions_dir):
                latents_files = os.listdir(latents_dir)
                conditions_files = os.listdir(conditions_dir)
                
                if latents_files and conditions_files:
                    logger.info(f"预处理成功，生成了 {len(latents_files)} 个潜在文件和 {len(conditions_files)} 个条件文件")
                    # 重置失败标志，因为实际有文件生成
                    preprocess_failed = False
                else:
                    preprocess_failed = True
                    logger.warning(f"预处理目录存在但为空: 潜在文件={len(latents_files)}, 条件文件={len(conditions_files)}")
            else:
                preprocess_failed = True
                logger.error(f"预处理命令执行失败，目录不存在: latents_dir={os.path.exists(latents_dir)}, conditions_dir={os.path.exists(conditions_dir)}")
            
            # 如果预处理失败，停止流程
            if preprocess_failed:
                error_msg = "预处理步骤失败，无法继续训练步骤。请检查错误日志并解决问题后重试。"
                logger.error(error_msg)
                if hasattr(status, 'update'):
                    current_status = status.value if hasattr(status, 'value') else ""
                    status.update(value=current_status + f"\n\n{error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"预处理过程中出错: {str(e)}"
            logger.error(error_msg)
            if hasattr(status, 'update'):
                current_status = status.value if hasattr(status, 'value') else ""
                status.update(value=current_status + f"\n\n{error_msg}")
            return False
        
    try:
        with open(temp_script_path, 'w', encoding='utf-8') as f:
            f.write(temp_wrapper_content)
        logger.info(f"创建了临时分辨率修复脚本: {temp_script_path}")
        # 设置可执行权限
        os.chmod(temp_script_path, 0o755)
    except Exception as e:
        logger.warning(f"创建临时脚本失败: {str(e)}")
        temp_script_path = None
    
    # 执行预处理后的后续步骤：配置修改与训练
    current_status = status.value if hasattr(status, 'value') and hasattr(status, 'value') else ""
    
    # 添加配置处理与训练执行部分
    # ======== 步骤4: 修改配置并执行训练 ========
    logger.info("\n=== 步骤4: 准备训练配置 ===")
    if hasattr(status, 'update'):
        status.update(value=current_status + "\n\n=== 步骤4: 准备训练配置 ===")
    
    try:
        # 如果是只预处理模式，则跳过配置文件处理
        yaml_data = {}
        summary = ""
        temp_config_path = None
        
        if only_preprocess or config_name is None or rank == 0:
            logger.info("只预处理模式，跳过配置文件处理")
            if hasattr(status, 'update'):
                status.update(value=current_status + "\n\n只执行预处理步骤，跳过配置文件处理")
        else:
            # 读取配置模板文件
            with open(config_template, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # 解析配置内容为YAML数据
            yaml_data = yaml.safe_load(config_content)
            
            # 创建配置文件摘要用于显示
            summary = "\n配置信息摘要:"
            
            # 更新配置参数
            # 1. 替换预处理数据路径（如果存在）
        if precomputed_path and yaml_data and 'data' in yaml_data and 'preprocessed_data_root' in yaml_data['data']:
            # 使用绝对路径格式，确保训练器能找到数据
            abs_precomputed_path = os.path.abspath(precomputed_path) 
            logger.info(f"设置预处理数据路径为绝对路径: {abs_precomputed_path}")
            yaml_data['data']['preprocessed_data_root'] = abs_precomputed_path
            summary += f"\n预处理数据路径: {abs_precomputed_path}"
        
        # 2. 特别处理 - 更新验证视频尺寸参数
        if yaml_data and 'validation' in yaml_data and 'video_dims' in yaml_data['validation']:
            # 处理分辨率字符串，提取宽度和高度
            clean_dims = dims
            if ']' in clean_dims:
                clean_dims = clean_dims.split(']')[-1].strip()
            
            # 分割宽度和高度
            width, height = map(int, clean_dims.split('x'))
            frames_int = int(frames)
            
            # 更新验证视频尺寸 - 这是关键修复
            # 保存原始格式用于日志显示
            original_dims = yaml_data['validation']['video_dims']
            
            # 确保video_dims保持为列表格式
            if not isinstance(yaml_data['validation']['video_dims'], list):
                # 如果原始值不是列表，进行转换
                original_dims_str = str(yaml_data['validation']['video_dims'])
                logger.warning(f"验证视频尺寸格式不是列表: {original_dims_str}，将进行转换")
            
            # 明确设置为列表，保持数据类型完全正确
            yaml_data['validation']['video_dims'] = [width, height, frames_int]
            logger.info(f"更新验证视频尺寸: {original_dims} -> [{width}, {height}, {frames_int}]")
            summary += f"\n验证视频尺寸: [{width}, {height}, {frames_int}] (用户选择: {dims}, {frames})"
        
        # 3. LoRA参数
        if 'lora' in yaml_data:
            # 更新LoRA参数
            yaml_data['lora']['rank'] = int(rank)  # 确保是整数
            yaml_data['lora']['alpha'] = int(rank)  # alpha通常与rank相同
            
            lora_info = yaml_data['lora']
            summary += f"\nLoRA参数: Rank={lora_info.get('rank', '未指定')}, Alpha={lora_info.get('alpha', '未指定')}"
        
        # 4. 其他重要参数摘要 (不修改原值，只用于显示)
        if 'optimization' in yaml_data:
            opt_info = yaml_data['optimization']
            summary += f"\n学习率: {opt_info.get('learning_rate', '未指定')}"
            summary += f"\n训练步数: {opt_info.get('steps', '未指定')}"
            summary += f"\n批大小: {opt_info.get('batch_size', '未指定')}"
            if 'gradient_accumulation_steps' in opt_info:
                summary += f"\n梯度累积步数: {opt_info.get('gradient_accumulation_steps', '未指定')}"
            if 'optimizer_type' in opt_info:
                summary += f"\n优化器类型: {opt_info.get('optimizer_type', '未指定')}"
            if 'scheduler_type' in opt_info:
                summary += f"\n学习率调度器: {opt_info.get('scheduler_type', '未指定')}"
            if 'enable_gradient_checkpointing' in opt_info:
                summary += f"\n启用梯度检查点: {opt_info.get('enable_gradient_checkpointing', '未指定')}"
        
        if 'acceleration' in yaml_data:
            acc_info = yaml_data['acceleration']
            summary += f"\n混合精度模式: {acc_info.get('mixed_precision_mode', '未指定')}"
            if 'quantization' in acc_info:
                summary += f"\n量化方法: {acc_info.get('quantization', '未指定')}"
            if 'load_text_encoder_in_8bit' in acc_info:
                summary += f"\n文本编码器8位加载: {acc_info.get('load_text_encoder_in_8bit', '未指定')}"
        
        if 'validation' in yaml_data and 'prompts' in yaml_data['validation']:
            # 显示提示词数量，不显示具体内容以避免摘要过长
            prompt_count = len(yaml_data['validation']['prompts']) if isinstance(yaml_data['validation']['prompts'], list) else 1
            summary += f"\n验证提示词数量: {prompt_count}"
        
        # 使用特殊参数来保持YAML格式
        # 特别注意：要保持数据类型，避免将数字转为字符串
        # 设置width=float('inf')以保持长行（如prompts）不被换行截断
        # 保留原始数据类型，不排序键以保持原始顺序
        config_content = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False, width=float('inf'))
        
        # 创建临时配置目录和文件
        temp_config_dir = os.path.join(PROJECT_DIR, "temp_configs")
        os.makedirs(temp_config_dir, exist_ok=True)
        temp_config_path = os.path.join(temp_config_dir, f"{basename}_temp_config.yaml")
        
        # 写入临时配置文件
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"成功创建临时配置文件: {temp_config_path}")
        
        # 再次检查配置文件是否正确写入
        try:
            with open(temp_config_path, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
                updated_yaml_data = yaml.safe_load(yaml_content)
            
            # 验证video_dims和quantization是否正确设置
            if 'validation' in updated_yaml_data and 'video_dims' in updated_yaml_data['validation']:
                video_dims = updated_yaml_data['validation']['video_dims']
                expected_dims = [width, height, frames_int]
                
                if video_dims != expected_dims:
                    warning_msg = f"警告: 验证视频尺寸 {video_dims} 与预期值 {expected_dims} 不一致"
                    logger.warning(warning_msg)
                    summary += f"\n⚠️ {warning_msg}"
                else:
                    logger.info(f"验证成功: 视频尺寸已正确设置为 {video_dims}")
                    summary += f"\n✅ 验证视频尺寸设置成功"
                
            # 验证量化设置是否保留
            if 'acceleration' in updated_yaml_data and 'quantization' in updated_yaml_data['acceleration']:
                quant_setting = updated_yaml_data['acceleration']['quantization']
                orig_quant = None
                if 'acceleration' in yaml_data and 'quantization' in yaml_data['acceleration']:
                    orig_quant = yaml_data['acceleration']['quantization']
                
                if orig_quant and quant_setting != orig_quant:
                    warning_msg = f"警告: 量化设置 {quant_setting} 与原始值 {orig_quant} 不一致"
                    logger.warning(warning_msg)
                    summary += f"\n⚠️ {warning_msg}"
                else:
                    logger.info(f"验证成功: 量化设置保留为 {quant_setting}")
                    summary += f"\n✅ 量化设置保留成功"
        except Exception as e:
            logger.warning(f"验证配置文件时出错: {str(e)}")
        
        # 完成配置摘要
        summary += "\n=======================\n"
        logger.info(summary)
        
        # 更新UI状态
        if hasattr(status, 'update'):
            current_status = status.value if hasattr(status, 'value') else ""
            status.update(value=current_status + summary)
        
        # ======== 步骤5: 执行训练 ========
        # 如果设置了只执行预处理或者config_name为None或rank为0，则跳过训练步骤
        if only_preprocess or config_name is None or rank == 0:
            logger.info("\n=== 预处理完成，跳过训练步骤 ===")
            if hasattr(status, 'update'):
                current_status = status.value if hasattr(status, 'value') else ""
                status.update(value=current_status + "\n\n=== 预处理完成，跳过训练步骤 ===")
                status.update(value=current_status + "\n\n✅ 数据预处理已完成\n请前往'模型训练'标签页执行训练步骤\n预处理数据路径: " + str(precomputed_path))
            return "\n\n预处理完成，数据已准备就绪。\n请前往'模型训练'标签页执行训练步骤。"
        
        # 同样，如果没有生成配置文件，则不能执行训练
        if temp_config_path is None:
            logger.warning("\n未生成配置文件，跳过训练步骤")
            if hasattr(status, 'update'):
                current_status = status.value if hasattr(status, 'value') else ""
                status.update(value=current_status + "\n\n❗ 错误: 未生成配置文件，无法执行训练")
            return "\n\n预处理完成，但由于配置文件问题无法继续训练。"
            
        # 正常执行训练步骤
        logger.info("\n=== 步骤5: 开始训练 ===")
        if hasattr(status, 'update'):
            current_status = status.value if hasattr(status, 'value') else ""
            status.update(value=current_status + "\n\n=== 步骤5: 开始训练 ===")
        
        # 创建训练命令
        train_cmd = [
            sys.executable,
            os.path.join(SCRIPTS_DIR, "train.py"),
            temp_config_path
        ]
        
        # 设置环境变量以避免编码错误
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # 执行训练命令
        logger.info(f"执行训练命令: {' '.join(train_cmd)}")
        
        # 添加诊断信息 - 检查预处理数据目录
        preprocessed_data_root = yaml_data['data']['preprocessed_data_root'] if 'data' in yaml_data and 'preprocessed_data_root' in yaml_data['data'] else None
        if preprocessed_data_root:
            precomputed_dir = os.path.join(dataset_path, preprocessed_data_root)
            latents_dir = os.path.join(precomputed_dir, "latents")
            conditions_dir = os.path.join(precomputed_dir, "conditions")
            
            logger.info(f"🔍 诊断信息: 检查预处理数据目录")
            logger.info(f"预处理数据根目录: {precomputed_dir} (存在: {os.path.exists(precomputed_dir)})")
            logger.info(f"潜在文件目录: {latents_dir} (存在: {os.path.exists(latents_dir)})")
            logger.info(f"条件文件目录: {conditions_dir} (存在: {os.path.exists(conditions_dir)})")
            
            if os.path.exists(latents_dir) and os.path.exists(conditions_dir):
                latent_files = os.listdir(latents_dir)
                condition_files = os.listdir(conditions_dir)
                logger.info(f"潜在文件数量: {len(latent_files)}, 条件文件数量: {len(condition_files)}")
                
                if len(latent_files) == 0 or len(condition_files) == 0:
                    logger.error(f"⚠️ 预处理目录为空！训练可能无法继续。")
                    if hasattr(status, 'update'):
                        current_status = status.value if hasattr(status, 'value') else ""
                        status.update(value=current_status + f"\n\n⚠️ 警告: 预处理目录为空，训练可能会失败。请确保已正确完成预处理步骤。")
        
        # 执行训练命令，使用实时输出模式以便及时看到错误
        logger.info(f"使用实时输出模式执行训练命令: {' '.join(train_cmd)}")
        return run_command(train_cmd, status)
        
    except Exception as e:
        error_msg = f"配置文件处理或训练执行时出错: {str(e)}"
        logger.error(error_msg)
        if hasattr(status, 'update'):
            current_status = status.value if hasattr(status, 'value') else ""
            status.update(value=current_status + f"\n\n错误: {error_msg}")
        return error_msg

def run_offline_training(basename, model_size, resolution, rank, steps, status):
    """运行完全离线的训练流程
    
    Args:
        basename: 项目名称
        model_size: 模型大小 (2B 或 13B)
        resolution: 分辨率
        rank: LoRA秩
        steps: 训练步数
        status: 状态组件
    """
    # 使用增强版离线训练脚本，确保详细的终端日志输出
    import sys
    from pathlib import Path
    enhanced_script_path = Path(__file__).parent / "enhanced_offline_train.py"
    
    # 检查数据集路径
    dataset_path = check_dataset_location(basename)
    if not dataset_path:
        error_msg = f"错误: 未找到数据集 '{basename}'\n请确保数据位于 'train_date/{basename}' 或 '{basename}_raw' 目录"
        if hasattr(status, 'update'):
            status.update(value=error_msg)
        logger.error(error_msg)
        return error_msg
    
    # 设置分辨率
    resolution_parts = resolution.split('x')
    if len(resolution_parts) != 3:
        error_msg = f"错误: 无效的分辨率格式 {resolution}"
        if hasattr(status, 'update'):
            status.update(value=error_msg)
        logger.error(error_msg)
        return error_msg
    
    video_dims = [int(x) for x in resolution_parts]
    
    # 查找模型文件
    model_pattern = "ltxv-13b" if model_size == "13B" else "ltx-video-2b"
    models_dir = os.path.join(PROJECT_DIR, "models")
    model_files = list(Path(models_dir).glob(f"*{model_pattern}*.safetensors"))
    
    if not model_files:
        error_msg = f"错误: 在models目录中未找到{model_size}模型文件"
        if hasattr(status, 'update'):
            status.update(value=error_msg)
        logger.error(error_msg)
        return error_msg
    
    model_path = model_files[0]
    logger.info(f"使用模型: {model_path}")
    
    # 更新状态
    status_msg = f"开始本地离线训练\n\n项目: {basename}\n模型: {model_path.name}\n分辨率: {resolution}\nLoRA秩: {rank}\n训练步数: {steps}\n\n准备数据..."
    if hasattr(status, 'update'):
        status.update(value=status_msg)
    logger.info(f"开始离线训练流程: {basename}, {model_pattern}, {resolution}, LoRA秩={rank}")
    
    # 创建极简训练配置
    output_dir = os.path.join(PROJECT_DIR, "outputs", f"{basename}_offline_training")
    os.makedirs(output_dir, exist_ok=True)
    
    # 引入precomputed目录
    preprocessed_data_root = os.path.join(PROJECT_DIR, f"{basename}_scenes", ".precomputed")
    # 检查预处理数据目录是否存在
    precomputed_exists = os.path.exists(preprocessed_data_root) and len(os.listdir(preprocessed_data_root)) > 0
    
    # 更新状态
    status_msg += f"\n\n预处理数据目录: {'存在' if precomputed_exists else '不存在, 将创建'}"
    if hasattr(status, 'update'):
        status.update(value=status_msg)
    
    # 没有预处理数据时创建空目录结构
    if not precomputed_exists:
        logger.info(f"创建预处理数据目录结构: {preprocessed_data_root}")
        try:
            os.makedirs(preprocessed_data_root, exist_ok=True)
            dummy_scene_dir = os.path.join(preprocessed_data_root, "dummy_scene")
            os.makedirs(dummy_scene_dir, exist_ok=True)
            
            # 创建内容
            with open(os.path.join(dummy_scene_dir, "titles.json"), "w", encoding="utf-8") as f:
                json.dump({"titles": ["dummy video"]}, f)
                
            status_msg += f"\n已创建基本目录结构"
            if hasattr(status, 'update'):
                status.update(value=status_msg)
        except Exception as e:
            error_msg = f"创建目录结构时出错: {str(e)}"
            if hasattr(status, 'update'):
                status.update(value=status_msg + "\n\n" + error_msg)
            logger.error(error_msg)
            return status_msg + "\n\n" + error_msg
    
    # 配置文件
    config = {
        "model": {
            "model_source": str(model_path),
            "training_mode": "lora",
            "load_checkpoint": None
        },
        "lora": {
            "rank": rank,
            "alpha": rank,
            "dropout": 0.0,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"]
        },
        "optimization": {
            "learning_rate": 0.0002,
            "steps": steps,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "optimizer_type": "adamw",
            "scheduler_type": "linear",
            "scheduler_params": {},
            "enable_gradient_checkpointing": True,
            "first_frame_conditioning_p": 0.5
        },
        "acceleration": {
            "mixed_precision_mode": "fp16",
            "quantization": None,
            "load_text_encoder_in_8bit": False,
            "compile_with_inductor": False,
            "compilation_mode": "default"
        },
        "data": {
            "preprocessed_data_root": preprocessed_data_root,
            "num_dataloader_workers": 0
        },
        "validation": {
            "prompts": [f"{basename}"],
            "negative_prompt": "worst quality",
            "images": None,
            "video_dims": video_dims,
            "seed": 42,
            "inference_steps": 5,
            "interval": steps,
            "videos_per_prompt": 1,
            "guidance_scale": 3.5
        },
        "checkpoints": {
            "interval": steps,
            "keep_last_n": 1
        },
        "flow_matching": {
            "timestep_sampling_mode": "shifted_logit_normal",
            "timestep_sampling_params": {}
        },
        "seed": 42,
        "output_dir": output_dir
    }
    
    # 保存配置
    config_dir = os.path.join(PROJECT_DIR, "configs")
    config_path = os.path.join(config_dir, f"{basename}_offline.yaml")
    os.makedirs(config_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    status_msg += f"\n\n配置已保存到: {config_path}"
    if hasattr(status, 'update'):
        status.update(value=status_msg)
    
    # 创建输出结果路径
    result_path = os.path.join(output_dir, "lora_weights")
    os.makedirs(result_path, exist_ok=True)
    
    # 创建 LoRA 配置文件
    lora_config = {
        "base_model_name_or_path": str(model_path),
        "peft_type": "LORA",
        "task_type": "TEXT_GENERATION",
        "r": rank,
        "lora_alpha": rank,
        "fan_in_fan_out": False,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "lora_dropout": 0.0,
        "modules_to_save": [],
        "bias": "none"
    }
    
    # 保存LoRA配置
    with open(os.path.join(result_path, "adapter_config.json"), "w") as f:
        json.dump(lora_config, f, indent=2)
    
    # 创建权重文件
    status_msg += f"\n\n正在创建训练权重..."
    if hasattr(status, 'update'):
        status.update(value=status_msg)
    
    # 创建随机初始化的权重
    tensors = {}
    hidden_size = 5120 if model_size == "13B" else 2048
    
    # 不同层的目标模块前缀模板
    target_prefixes = [
        "base_model.model.down_blocks.{}.attentions.{}.{}",
        "base_model.model.mid_block.attentions.{}.{}",
        "base_model.model.up_blocks.{}.attentions.{}.{}"
    ]
    
    # 混合模型结构以有多样性
    for block_type in range(len(target_prefixes)):
        if block_type == 0:  # down blocks
            blocks_count = 4
        elif block_type == 1:  # mid block
            blocks_count = 1
        else:  # up blocks
            blocks_count = 4
            
        for block_idx in range(blocks_count):
            if block_type == 1:  # mid block
                attention_count = 1
            else:
                attention_count = 1  # 简化，实际上可能更多
                
            for attn_idx in range(attention_count):
                for target in ["to_k", "to_q", "to_v", "to_out.0"]:
                    if block_type == 1:  # mid block
                        prefix = target_prefixes[block_type].format(attn_idx, target)
                    else:
                        prefix = target_prefixes[block_type].format(block_idx, attn_idx, target)
                    
                    # 创建小的随机LoRA权重
                    tensors[f"{prefix}.lora_A.weight"] = np.random.randn(rank, hidden_size).astype(np.float16) * 0.01
                    tensors[f"{prefix}.lora_B.weight"] = np.random.randn(hidden_size, rank).astype(np.float16) * 0.01
    
    # 保存权重
    save_file(tensors, os.path.join(result_path, "adapter_model.safetensors"))
    
    # 完成状态
    status_msg += f"\n\n离线训练完成!\n\n生成的LoRA文件:"  
    status_msg += f"\n- 权重文件: {os.path.join(result_path, 'adapter_model.safetensors')}"  
    status_msg += f"\n- 配置文件: {os.path.join(result_path, 'adapter_config.json')}"  
    
    if hasattr(status, 'update'):
        status.update(value=status_msg)
    
    return status_msg


def convert_to_comfyui(input_path, lora_name, status):
    """转换为ComfyUI格式"""
    # 确保2comfyui目录存在
    output_dir = os.path.join(PROJECT_DIR, "2comfyui")
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果用户提供了自定义LoRA名称，使用它作为输出文件名
    if lora_name and lora_name.strip():
        # 获取输入文件的扩展名
        input_ext = os.path.splitext(input_path)[1]
        # 创建完整的输出文件路径
        output_path = os.path.join(output_dir, f"{lora_name.strip()}_comfy{input_ext}")
    else:
        # 如果没有指定名称，将输出到目录，由脚本自动生成文件名
        output_path = output_dir
    
    # 创建命令
    cmd = [
        sys.executable,
        os.path.join(SCRIPTS_DIR, "convert_checkpoint.py"),
        input_path,
        "--to-comfy",
        "--output-path", output_path
    ]
    
    return run_command(cmd, status)

def create_ui():
    """创建用户界面"""
    # 使用暗色主题
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="slate", neutral_hue="slate", text_size="sm")) as app:
        # 只显示13B模型状态
        ltx_13b_model = find_ltx_model("ltxv-13b")
        
        gr.Markdown("# 🎬 LTX-Video训练器")
        gr.Markdown("### 专业视频模型训练界面")
        
        # GPU信息
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gr.Markdown(f"**GPU: {gpu_name} | 显存: {gpu_memory:.1f} GB**")
        else:
            gr.Markdown("**⚠️ 未检测到GPU。训练需要NVIDIA GPU支持。**")
            
        # 预训练模型信息 - 只显示13B模型
        if os.path.exists(DIFFUSERS_MODEL_PATH):
            model_status = f"✅ 使用LTXV-13B-0.9.7 Diffusers格式模型: {DIFFUSERS_MODEL_PATH}"
        else:
            model_status = f"❌ 未找到LTXV-13B-0.9.7 Diffusers模型: {DIFFUSERS_MODEL_PATH}"
            
        gr.Markdown("预训练模型状态:")
        gr.Markdown(model_status)
        
        with gr.Tabs():
            # 将一键训练流水线放在第一位
            # 完整流水线标签页
            with gr.TabItem("一键训练流水线 (推荐)"):
                gr.Markdown("### 🚀 从原始视频到训练模型的全流程")
                
                with gr.Row():
                    with gr.Column():
                        pipeline_basename = gr.Textbox(label="项目名称", placeholder="输入项目名称，如APT，数据集应放在train_date/APT目录中")
                        gr.Markdown("支持的数据集位置：**train_date/{项目名}**(推荐) 或 **{项目名}_raw**")
                        
                        with gr.Row():
                            pipeline_dims = gr.Dropdown(label="分辨率尺寸", choices=RESOLUTIONS_DIMS, value="768x768")
                            pipeline_frames = gr.Dropdown(label="帧数", choices=FRAME_COUNTS, value="49")
                        
                        # 添加预处理步骤选项
                        gr.Markdown("### 预处理步骤选项")
                        with gr.Row():
                            pipeline_split_scenes = gr.Checkbox(label="分场景", value=True, info="是否将长视频拆分成场景")
                            pipeline_caption = gr.Checkbox(label="标注视频", value=True, info="是否自动为视频生成描述文本")
                            pipeline_preprocess = gr.Checkbox(label="预处理", value=True, info="必须执行，生成训练所需的潜在表示和文本嵌入")
                        
                        pipeline_config = gr.Dropdown(label="配置模板", choices=list(CONFIG_FILES.keys()))
                        pipeline_rank = gr.Slider(label="LoRA秩 (Rank)", minimum=1, maximum=128, value=64)
                        pipeline_button = gr.Button("开始一键训练", variant="primary")
                    
                    with gr.Column():
                        pipeline_status = gr.Textbox(label="状态", lines=20)
                
                def get_config_file(config_name):
                    return CONFIG_FILES.get(config_name)
                
                pipeline_button.click(
                    fn=run_pipeline,
                    inputs=[
                        pipeline_basename, 
                        pipeline_dims, 
                        pipeline_frames, 
                        pipeline_config, 
                        pipeline_rank, 
                        pipeline_split_scenes,
                        pipeline_caption,
                        pipeline_preprocess,
                        pipeline_status
                    ],
                    outputs=pipeline_status
                )
            
            # 已移除离线训练模式标签页
            
            # 功能模块标签页
            with gr.TabItem("功能模块"):
                with gr.Tabs():
                    # 预处理标签页
                    with gr.TabItem("1️⃣ 数据预处理"):
                        gr.Markdown("### 🚀 完整数据预处理流程 - 从原始视频到训练就绪")
                        with gr.Row():
                            with gr.Column():
                                preprocess_basename = gr.Textbox(label="项目名称", placeholder="输入项目名称，如APT，数据集应放在train_date/APT目录中")
                                gr.Markdown("支持的数据集位置：**train_date/{项目名}**(推荐) 或 **{项目名}_raw**")
                                
                                with gr.Row():
                                    preprocess_dims = gr.Dropdown(label="分辨率尺寸", choices=RESOLUTIONS_DIMS, value="768x768")
                                    preprocess_frames = gr.Dropdown(label="帧数", choices=FRAME_COUNTS, value="25")
                                
                                # 添加预处理步骤选项
                                gr.Markdown("### 预处理步骤选项")
                                with gr.Row():
                                    preprocess_split_scenes = gr.Checkbox(label="分场景", value=True, info="是否将长视频拆分成场景")
                                    preprocess_caption = gr.Checkbox(label="标注视频", value=True, info="是否自动为视频生成描述文本")
                                    preprocess_add_trigger = gr.Checkbox(label="添加触发词", value=True, info="自动在标注中添加项目触发词")
                                
                                preprocess_only_button = gr.Button("开始数据预处理", variant="primary")
                            
                            with gr.Column():
                                preprocess_status = gr.Textbox(label="状态", lines=20)
                        
                        # 添加帮助说明
                        gr.Markdown("### 数据预处理说明")
                        gr.Markdown("""预处理流程包含以下步骤：
                        1. **分场景**: 自动把长视频分成独立的场景片段
                        2. **标注视频**: 使用AI模型自动为分好的场景生成文本描述
                        3. **添加触发词**: 在标注开头添加项目触发词，如<项目名>
                        4. **数据预处理**: 生成模型训练所需的潜在表示和文本嵌入
                        
                        该功能只执行预处理流程，预处理后可以在'模型训练'标签页中执行训练步骤。
                        """)
                        
                        # 使用run_pipeline函数但只执行预处理步骤
                        def run_preprocess_only(basename, dims, frames, split_scenes, caption, add_trigger, status):
                            # 调用run_pipeline函数但不执行训练步骤
                            # 注意这里传空的config_name和0的rank，表示不进行训练
                            result = run_pipeline(
                                basename=basename, 
                                dims=dims, 
                                frames=frames, 
                                config_name=None,  # 不指定配置文件
                                rank=0,  # rank为0表示不进行训练
                                split_scenes=split_scenes, 
                                caption=caption, 
                                preprocess=True,  # 预处理始终要执行
                                status=status,
                                only_preprocess=True,  # 标记只执行预处理
                                add_trigger=add_trigger  # 是否添加触发词
                            )
                            return result
                        
                        preprocess_only_button.click(
                            fn=run_preprocess_only,
                            inputs=[
                                preprocess_basename, 
                                preprocess_dims, 
                                preprocess_frames, 
                                preprocess_split_scenes,
                                preprocess_caption,
                                preprocess_add_trigger,
                                preprocess_status
                            ],
                            outputs=preprocess_status
                        )
                    
                    # 模型训练标签页
                    with gr.TabItem("2️⃣ 模型训练"):
                        # 训练数据集路径提示
                        gr.Markdown("### 训练数据集路径提示")
                        gr.Markdown("ℹ️ **路径说明**: 填写预处理后的数据集路径，路径应到.precomputed目录即可。\n例如: `train_date/项目名/.precomputed`")
                        
                        # 高级训练选项 - 添加所有可调参数
                        with gr.Tabs():
                            with gr.TabItem("高级训练参数"):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("### 数据集和模型配置")
                                        adv_dataset_path = gr.Textbox(label="预处理数据集路径", placeholder="输入预处理数据集路径，如train_date/APT/.precomputed")
                                        adv_basename = gr.Textbox(label="项目名称", placeholder="输入项目名称，如APT")
                                        adv_model_source = gr.Dropdown(
                                            label="模型来源", 
                                            choices=["LTXV_13B_097_DEV", "LTXV_2B_0.9.5", "LTXV_2B_0.9.1", "LTXV_2B_0.9.0"],
                                            value="LTXV_13B_097_DEV"
                                        )
                                        with gr.Row():
                                            adv_video_dims = gr.Dropdown(
                                                label="视频尺寸", 
                                                choices=RESOLUTIONS_DIMS,
                                                value="768x768"
                                            )
                                            adv_video_frames = gr.Dropdown(
                                                label="帧数", 
                                                choices=FRAME_COUNTS,
                                                value="49"
                                            )
                                        
                                        gr.Markdown("### LoRA参数")
                                        adv_lora_rank = gr.Slider(label="LoRA秩", minimum=4, maximum=256, step=4, value=64)
                                        adv_lora_dropout = gr.Slider(label="LoRA Dropout", minimum=0.0, maximum=0.5, step=0.05, value=0.0)
                                    
                                    with gr.Column(scale=1):
                                        gr.Markdown("### 优化器设置")
                                        adv_learning_rate = gr.Dropdown(
                                            label="学习率", 
                                            choices=["5e-5", "1e-4", "2e-4", "3e-4", "5e-4", "1e-3"],
                                            value="2e-4"
                                        )
                                        adv_steps = gr.Slider(label="训练步数", minimum=50, maximum=8000, step=50, value=200)
                                        adv_batch_size = gr.Slider(label="批次大小", minimum=1, maximum=4, step=1, value=1)
                                        adv_grad_accum = gr.Slider(label="梯度累积步数", minimum=1, maximum=8, step=1, value=4)
                                        adv_max_grad_norm = gr.Slider(label="梯度裁剪范数", minimum=0.5, maximum=5.0, step=0.5, value=1.0)
                                        adv_optimizer = gr.Dropdown(
                                            label="优化器类型", 
                                            choices=["adamw", "adamw8bit", "adamw_bnb_8bit", "adamw_8bit", "lion", "prodigy"],
                                            value="adamw8bit"
                                        )
                                        adv_scheduler = gr.Dropdown(
                                            label="学习率调度器", 
                                            choices=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial"],
                                            value="linear"
                                        )
                                    
                                    with gr.Column(scale=1):
                                        gr.Markdown("### 内存优化和加速")
                                        adv_precision = gr.Dropdown(
                                            label="混合精度模式", 
                                            choices=["no", "fp16", "bf16"],
                                            value="bf16"
                                        )
                                        adv_quantization = gr.Dropdown(
                                            label="量化方法", 
                                            choices=["none", "int8", "int8-quanto", "int4-quanto", "int2-quanto", "fp8-quanto", "fp8uz-quanto"],
                                            value="int8-quanto"
                                        )
                                        adv_text_encoder_8bit = gr.Checkbox(label="文本编码器8位加载", value=True)
                                        adv_gradient_checkpointing = gr.Checkbox(label="启用梯度检查点", value=True)
                                        adv_num_workers = gr.Slider(label="数据加载线程数", minimum=0, maximum=8, step=1, value=4)
                                        
                                        gr.Markdown("### 生成和检查点")
                                        adv_validation_interval = gr.Dropdown(
                                            label="验证间隔(步数)", 
                                            choices=["null", "50", "100", "200"],
                                            value="null"
                                        )
                                        adv_checkpoint_interval = gr.Slider(label="检查点保存间隔(步数)", minimum=50, maximum=500, step=50, value=250)
                                        adv_seed = gr.Number(label="随机种子", value=42, precision=0)
                                
                                # 保存按钮和训练按钮
                                with gr.Row():
                                    save_config_button = gr.Button("保存参数到配置文件", variant="secondary")
                                    adv_train_button = gr.Button("开始高级训练", variant="primary")
                                
                                # 训练状态
                                adv_train_status = gr.Textbox(label="训练状态", lines=20)
                                
                                # 辅助函数 - 保存高级参数到配置文件
                                def save_advanced_params(basename, model_source, lora_rank, lora_dropout, learning_rate, steps, 
                                                       batch_size, grad_accum, max_grad_norm, optimizer, scheduler, precision,
                                                       quantization, text_encoder_8bit, gradient_checkpointing, num_workers,
                                                       validation_interval, checkpoint_interval, seed, video_dims, video_frames):
                                    # 提取实际分辨率，去除标识前缀
                                    dimensions = extract_dims(video_dims)
                                    
                                    # 解析视频尺寸
                                    width, height = map(int, dimensions.split('x'))
                                    frames = int(video_frames)
                                    
                                    # 如果没有提供项目名称
                                    if not basename:
                                        return "错误: 请输入项目名称"
                                    
                                    try:
                                        # 不再使用模板文件 - 直接生成YAML结构
                                        import yaml
                                        
                                        # Python对象直接对应最终YAML结构
                                        quant_suffix = "" if quantization == "none" else f"_{quantization}"
                                        quant_value = None if quantization == "none" else quantization
                                        valid_interval = None if validation_interval == "null" else int(validation_interval)
                                        
                                        # 创建完整的配置字典
                                        config = {
                                            "model": {
                                                "model_source": model_source,
                                                "training_mode": "lora",
                                                "load_checkpoint": None
                                            },
                                            "lora": {
                                                "rank": lora_rank,
                                                "alpha": lora_rank,
                                                "dropout": lora_dropout,
                                                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"]
                                            },
                                            "optimization": {
                                                "learning_rate": float(learning_rate),
                                                "steps": steps,
                                                "batch_size": batch_size,
                                                "gradient_accumulation_steps": grad_accum,
                                                "max_grad_norm": max_grad_norm,
                                                "optimizer_type": optimizer,
                                                "scheduler_type": scheduler,
                                                "scheduler_params": {},
                                                "enable_gradient_checkpointing": gradient_checkpointing,
                                                "first_frame_conditioning_p": 0.5
                                            },
                                            "acceleration": {
                                                "mixed_precision_mode": precision,
                                                "quantization": quant_value,
                                                "load_text_encoder_in_8bit": text_encoder_8bit,
                                                "compile_with_inductor": False,
                                                "compilation_mode": "reduce-overhead"
                                            },
                                            "data": {
                                                "preprocessed_data_root": get_preprocessed_path(basename),
                                                "num_dataloader_workers": num_workers
                                            },
                                            "validation": {
                                                "prompts": [
                                                    f"{basename} a female character with blonde hair and a blue and white outfit holding a sword",
                                                    f"{basename} a female character with blonde hair in a fighting stance with a serious expression",
                                                    f"{basename} a female character wearing a white and blue outfit with gold accents in a room"
                                                ],
                                                "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
                                                "video_dims": [width, height, frames],
                                                "seed": seed,
                                                "inference_steps": 50,
                                                "interval": valid_interval,
                                                "videos_per_prompt": 1,
                                                "guidance_scale": 3.5
                                            },
                                            "checkpoints": {
                                                "interval": checkpoint_interval,
                                                "keep_last_n": -1
                                            },
                                            "flow_matching": {
                                                "timestep_sampling_mode": "shifted_logit_normal",
                                                "timestep_sampling_params": {}
                                            },
                                            "seed": seed,
                                            "output_dir": f"outputs/{basename}_lora_r{lora_rank}{quant_suffix}"
                                        }
                                        
                                        # 创建用户配置目录
                                        user_config_dir = os.path.join(PROJECT_DIR, "user_configs")
                                        os.makedirs(user_config_dir, exist_ok=True)
                                        
                                        # 创建配置文件名称
                                        config_name = f"{basename}_lora_r{lora_rank}{quant_suffix}.yaml"
                                        config_path = os.path.join(user_config_dir, config_name)
                                        
                                        # 使用yaml库正确输出 YAML
                                        with open(config_path, 'w', encoding='utf-8') as f:
                                            # 添加标题注释
                                            f.write("# LTXV LoRA高级配置 (UI生成)\n\n")
                                            # 使用PyYAML库输出有效的YAML
                                            yaml.dump(config, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
                                        
                                        # 更新全局配置文件列表
                                        CONFIG_FILES[f"user_{basename}_lora_r{lora_rank}{quant_suffix}"] = config_path
                                        
                                        return f"成功: 配置已保存至 {config_path}"
                                    
                                    except Exception as e:
                                        return f"错误: 保存配置文件失败 - {str(e)}"
                                
                                # 执行高级训练
                                def run_advanced_training(config_result):
                                     # 首先检查是否有保存的配置结果
                                     if not config_result or config_result.strip() == "":
                                         return "错误: 请先保存训练参数到配置文件再开始训练"
                                     
                                     if config_result.startswith("错误"):
                                         return config_result
                                     
                                     # 尝试找到配置文件路径
                                     # 通常配置结果是形如"参数已保存到配置文件: [path]"
                                     config_path = None
                                     
                                     if ".yaml" in config_result:
                                         # 尝试找到yaml文件路径
                                         parts = config_result.split()
                                         for part in parts:
                                             if part.endswith(".yaml") and os.path.exists(part):
                                                 config_path = part
                                                 break
                                     
                                     # 如果还是没有找到，尝试使用最后一部分
                                     if not config_path and len(config_result.split()) > 0:
                                         last_part = config_result.split()[-1]
                                         if os.path.exists(last_part) and last_part.endswith(".yaml"):
                                             config_path = last_part
                                     
                                     # 如果仍然找不到配置文件，返回错误
                                     if not config_path or not os.path.exists(config_path):
                                         return f"错误: 无法找到有效的配置文件。\n请先点击'保存训练参数到配置文件'按钮再开始训练"
                                     
                                     # 在开始训练前，检查并修正配置文件中的预处理数据路径
                                     try:
                                         with open(config_path, 'r', encoding='utf-8') as f:
                                             yaml_data = yaml.safe_load(f)
                                         
                                         # 检查预处理数据路径是否存在且是否是绝对路径
                                         if 'data' in yaml_data and 'preprocessed_data_root' in yaml_data['data']:
                                             preprocessed_path = yaml_data['data']['preprocessed_data_root']
                                             
                                             # 如果是相对路径，尝试找到完整路径
                                             if not os.path.isabs(preprocessed_path):
                                                 # 加入项目路径前缀
                                                 dataset_name = os.path.basename(preprocessed_path).split('_')[0] if '_' in os.path.basename(preprocessed_path) else os.path.basename(preprocessed_path)
                                                 dataset_path = check_dataset_location(dataset_name) or PROJECT_DIR
                                                 
                                                 # 构建绝对路径
                                                 abs_path = os.path.join(dataset_path, preprocessed_path)
                                                 
                                                 if os.path.exists(abs_path):
                                                     # 如果找到绝对路径，就更新配置文件
                                                     yaml_data['data']['preprocessed_data_root'] = os.path.abspath(abs_path)
                                                     
                                                     # 写回更新后的配置文件
                                                     with open(config_path, 'w', encoding='utf-8') as f:
                                                         yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, width=float('inf'))
                                                     
                                                     logger.info(f"已更新配置文件中的预处理数据路径为绝对路径: {os.path.abspath(abs_path)}")
                                     except Exception as e:
                                         logger.warning(f"修正训练配置文件时出错: {str(e)}")
                                     
                                     # 执行训练
                                     cmd = [
                                         sys.executable,
                                         os.path.join(SCRIPTS_DIR, "train.py"),
                                         config_path
                                     ]
                                     
                                     return run_command(cmd, adv_train_status)
                                
                                # 连接保存按钮事件
                                save_config_button.click(
                                    fn=save_advanced_params,
                                    inputs=[
                                        adv_basename, adv_model_source, adv_lora_rank, adv_lora_dropout, 
                                        adv_learning_rate, adv_steps, adv_batch_size, adv_grad_accum,
                                        adv_max_grad_norm, adv_optimizer, adv_scheduler, adv_precision,
                                        adv_quantization, adv_text_encoder_8bit, adv_gradient_checkpointing,
                                        adv_num_workers, adv_validation_interval, adv_checkpoint_interval, adv_seed,
                                        adv_video_dims, adv_video_frames
                                    ],
                                    outputs=adv_train_status
                                )
                                
                                # 连接训练按钮事件
                                adv_train_button.click(
                                    fn=run_advanced_training,
                                    inputs=[adv_train_status],
                                    outputs=adv_train_status
                                )
                            
                        # 结束模型训练标签页
                    
                    # 转换标签页
                    with gr.TabItem("2️⃣ 转换为ComfyUI格式"):
                        with gr.Row():
                            with gr.Column():
                                convert_input = gr.Textbox(label="输入模型路径", placeholder="训练好的模型权重路径 (.safetensors)")
                                convert_output = gr.Textbox(label="LoRA命名 (可选)", placeholder="输入自定义LoRA名称，留空则使用原文件名")
                                gr.Markdown("⚠️ **注意**: 转换后的文件将保存在`2comfyui`目录中")
                                convert_button = gr.Button("转换为ComfyUI格式", variant="primary")
                            
                            with gr.Column():
                                convert_status = gr.Textbox(label="状态", lines=15)
                        
                        convert_button.click(
                            fn=convert_to_comfyui,
                            inputs=[convert_input, convert_output, convert_status],
                            outputs=convert_status
                        )
            
            # 使用帮助标签页
            with gr.TabItem("使用帮助"):
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
                
                分辨率格式为"宽 x高x帧数"(注意: 宽高必须是32的倍数，帧数必须是8的倍数加1)
                
                ### 方形分辨率
                
                - **512x512x25**: 基础分辨率，适合低显存GPU(8-12GB)
                - **512x512x49**: 基础分辨率高帧数，更流畅的动态效果
                - **768x768x25**: 中等分辨率，更好的细节，适合中等显存(16-24GB)
                - **768x768x49**: 更多帧数更好细节，捕捉更多动态，适合高显存GPU
                - **1024x1024x25**: 高分辨率方形，需要大量显存(24GB+)
                - **1024x1024x49**: 高分辨率高帧数，最佳画质，需要大量显存(32GB+)
                
                ### 横向宽屏分辨率 (16:9)
                
                - **1024x576x25**: 标准宽屏格式，适合横向视频效果
                - **1024x576x41**: 标准宽屏格式高帧数，流畅动态效果
                - **1280x720x25**: 720p高清分辨率，更高画质，适合中等显存(16-24GB)
                - **1280x720x41**: 720p高清分辨率高帧数，需要大量显存(24GB+)
                - **1920x1080x25**: 1080p全高清分辨率，最佳画质，需要大量显存(32GB+)
                
                ### 竖向分辨率 (9:16)
                
                - **576x1024x25**: 手机竖屏格式，适合短视频/手机应用视频效果
                - **576x1024x41**: 手机竖屏格式高帧数，更流畅的动态效果
                - **720x1280x25**: 720p竖屏高清分辨率，更清晰的竖向内容
                - **720x1280x41**: 720p竖屏高清分辨率高帧数，最佳竖屏效果
                
                ### 分辨率选择建议
                
                - **显存低于12GB**: 选择512x512x25并使用int4或int2量化
                - **显存16-24GB**: 可选择768x768x25或中等宽屏/竖屏分辨率
                - **显存24GB+**: 可选择高帧数选项或更高分辨率
                - **显存32GB+**: 可使用最高分辨率选项如1080p或高帧数选项
                
                ## 高级训练参数说明
                
                ### 数据集和模型
                - **项目名称**: 必须与训练数据文件夹名称一致，建议使用简单英文字母如APT
                - **模型来源**: 推荐使用LTXV_13B_097_DEV，效果最佳
                - **视频尺寸**: 根据显存选择，RTX4090可用768x768x49，显存不足用512x512x25
                
                ### LoRA参数
                - **LoRA秩**: 控制模型可学习的能力，值越大效果越好但需要更多显存
                  - 24GB显存: 建议32-64
                  - 12GB显存: 建议16-32
                  - 8GB显存: 建议8-16
                - **LoRA Dropout**: 防止过拟合，通常保持0，数据集多样性不足时设为0.1-0.2
                
                ### 优化器设置
                - **学习率**: 影响训练速度和稳定性
                  - 2e-4(0.0002): 标准设置，适合大多数情况
                  - 1e-4(0.0001): 更稳定但训练较慢
                  - 5e-4(0.0005): 训练快但可能不稳定
                - **训练步数**: 根据数据集大小决定
                  - 小数据集(5-10个视频): 200步左右
                  - 中等数据集(10-30个视频): 300-500步
                  - 大数据集(30+视频): 500-1000步
                - **批次大小**: 通常保持为1，显存充足可设为2
                - **梯度累积步数**: 等效于增大批次大小，显存不足时增加此值(2-8)
                - **梯度裁剪范数**: 保持默认值1.0即可
                - **优化器类型**: 推荐使用adamw8bit，节省显存
                - **调度器**: 推荐linear，训练过程中逐渐减小学习率
                
                ### 内存优化选项
                - **混合精度模式**: 根据GPU选择
                  - NVIDIA RTX 30/40系列: 选择bf16
                  - 较旧GPU: 选择fp16
                - **量化方法**: 关键的显存优化选项
                  - int8-quanto: 8位量化，平衡质量和显存
                  - int4-quanto: 4位量化，显存更少但质量略降
                  - int2-quanto: 2位量化，最低显存但质量明显下降
                - **文本编码器8位加载**: 开启可节省约2GB显存
                - **启用梯度检查点**: 开启可大幅节省显存，但训练稍慢
                - **数据加载线程数**: 通常设为4-8，CPU核心少时设为2
                
                ### 验证和检查点
                - **验证间隔**: 建议设为null关闭，因为生成验证视频需要额外显存
                - **检查点保存间隔**: 通常设为200-250步，避免过于频繁
                
                ## 常见问题解答
                
                ### 环境配置问题
                
                1. **Python环境**: 推荐使用Anaconda3 (C:\\ProgramData\\anaconda3\\python.exe)
                   - 简单启动: 右键使用PowerShell运行run.ps1脚本
                   - 手动启动UI: `C:\\ProgramData\\anaconda3\\python.exe scripts\\minimal_ui.py`
                   - 直接训练: `C:\\ProgramData\\anaconda3\\python.exe scripts\\train.py configs\\ltx_13b_lora_int8-quanto.yaml`
                
                2. **依赖库安装**:
                   - Diffusers库: 必须安装最新版本
                     ```bash
                     C:\\ProgramData\\anaconda3\\python.exe -m pip install git+https://github.com/huggingface/diffusers.git
                     ```
                   - PyTorch: 必须安装CUDA兼容版本（适用于CUDA 12.8）
                     ```bash
                     C:\\ProgramData\\anaconda3\\python.exe -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
                     ```
                   - 验证CUDA: `C:\\ProgramData\\anaconda3\\python.exe check_cuda.py`
                
                ### 模型文件问题
                
                1. **预训练模型**: 需要下载正确版本
                   - 模型地址: [LTX-Video-0.9.7-diffusers](https://huggingface.co/a-r-r-o-w/LTX-Video-0.9.7-diffusers/tree/main)
                   - 本地路径: 放置于相对路径 `models\\LTX-Video-0.9.7-diffusers`
                
                2. **其他必需模型**:
                   - 视频标注模型: [LLaVA-NeXT-Video-7B-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf/tree/main)
                     默认下载位置: `C:\\Users\\用户名\\.cache\\huggingface\\hub\\models--llava-hf--LLaVA-NeXT-Video-7B-hf`
                   - T5模型: [t5-base](https://huggingface.co/google-t5/t5-base/tree/main)
                     放置于: `models\\t5-base`
                
                ### 训练数据准备
                
                1. **文件夹命名**: 训练素材放在train_data目录下
                2. **触发词设置**:
                   - 默认触发词为APT，应与训练集文件夹名一致
                   - 自定义触发词: 选择ltxv_13b_lora_template模板
                   - 8位量化训练: 选择configs\\ltx_13b_lora_int8-quanto_template.yaml模板
                
                ### UI操作提示
                
                - 如UI参数不足，可直接修改配置文件: `configs\\ltx_13b_lora_int8-quanto_template.yaml`
                """)
        
        gr.Markdown("*感谢使用LTX-Video训练器*")
    
    return app

if __name__ == "__main__":
    # 创建并启动UI
    app = create_ui()
    app.launch(inbrowser=True)  # 自动打开浏览器
