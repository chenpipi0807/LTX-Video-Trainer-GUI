#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTX-Video-Trainer 高级训练参数模块
提供专业训练参数的定义和处理函数
"""

import os
import json
import logging
from pathlib import Path

# 设置日志
logger = logging.getLogger('LTX-Pro-Trainer.Params')

# 获取项目根目录
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_DIR, "config")

# 确保配置目录存在
os.makedirs(CONFIG_DIR, exist_ok=True)

# 默认高级训练参数定义
DEFAULT_TRAINING_PARAMS = {
    "model": {
        "model_source": "auto",
        "training_mode": "lora",
        "load_checkpoint": None
    },
    "lora": {
        "rank": 32,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "modules_to_save": []
    },
    "optimization": {
        "learning_rate": 0.0002,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 5,
        "lr_warmup_ratio": 0.05,
        "min_learning_rate": 0.00001,
        "steps": 50,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "optimizer_type": "adamw",
        "optimizer_params": {
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        },
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
        "preprocessed_data_root": "auto",
        "num_dataloader_workers": 0,
        "shuffle_batches": True,
        "image_augmentation": {
            "enabled": True,
            "horizontal_flip_p": 0.0,
            "vertical_flip_p": 0.0,
            "random_crop_p": 0.0,
            "random_rotation_p": 0.0,
            "color_jitter_p": 0.0
        }
    },
    "validation": {
        "prompts": ["project_name"],
        "negative_prompt": "worst quality",
        "images": None,
        "video_dims": [768, 768, 25],
        "seed": 42,
        "inference_steps": 25,
        "interval": 50,
        "videos_per_prompt": 1,
        "guidance_scale": 7.5
    },
    "checkpoints": {
        "interval": 50,
        "keep_last_n": 1
    },
    "flow_matching": {
        "timestep_sampling_mode": "shifted_logit_normal",
        "timestep_sampling_params": {
            "logit_mean": 0.0,
            "logit_stddev": 2.0,
            "min_value": 0.02,
            "max_value": 0.98
        }
    },
    "advanced_options": {
        "use_deepspeed": False,
        "deepspeed_stage": 2,
        "use_offload": False,
        "offload_to_cpu": False,
        "memory_efficient_attention": True,
        "torch_compile": False,
        "use_xformers": True,
        "train_text_encoder": False,
        "text_encoder_learning_rate": 0.00005
    },
    "seed": 42,
    "output_dir": "auto"
}

# 预设训练配置
TRAINING_PRESETS = {
    "超快速训练": {
        "lora": {"rank": 16, "dropout": 0.0},
        "optimization": {"steps": 20, "learning_rate": 0.0003},
        "validation": {"inference_steps": 5}
    },
    "标准训练": {
        "lora": {"rank": 32, "dropout": 0.05},
        "optimization": {"steps": 50, "learning_rate": 0.0002},
        "validation": {"inference_steps": 25}
    },
    "高质量训练": {
        "lora": {"rank": 64, "dropout": 0.1},
        "optimization": {"steps": 100, "learning_rate": 0.0001},
        "validation": {"inference_steps": 30}
    },
    "精细调整": {
        "lora": {"rank": 32, "dropout": 0.05},
        "optimization": {"steps": 50, "learning_rate": 0.00005, "lr_scheduler": "constant"},
        "validation": {"inference_steps": 30}
    },
    "低显存模式": {
        "lora": {"rank": 16, "dropout": 0.0},
        "optimization": {"gradient_accumulation_steps": 2, "enable_gradient_checkpointing": True},
        "acceleration": {"mixed_precision_mode": "fp16", "load_text_encoder_in_8bit": True},
        "advanced_options": {"memory_efficient_attention": True, "use_xformers": True}
    }
}

# 优化器选项
OPTIMIZER_OPTIONS = [
    "adamw", "adam", "sgd", "adafactor", "adagrad", "adamax"
]

# 学习率调度器选项
LR_SCHEDULER_OPTIONS = [
    "linear", "cosine", "cosine_with_restarts", "polynomial",
    "constant", "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau"
]

# 混合精度模式选项
MIXED_PRECISION_OPTIONS = [
    "no", "fp16", "bf16"
]

# 时间步采样模式选项
TIMESTEP_SAMPLING_OPTIONS = [
    "uniform", "truncated_normal", "normal", "logit_normal", "shifted_logit_normal"
]

def load_config(config_path):
    """加载JSON配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"读取配置文件失败: {str(e)}")
        return None

def save_config(config, config_path):
    """保存配置到JSON文件"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"保存配置文件失败: {str(e)}")
        return False

def get_config_files():
    """读取可用的配置文件"""
    config_files = {}
    try:
        if os.path.exists(CONFIG_DIR):
            for file in os.listdir(CONFIG_DIR):
                if file.endswith(".json"):
                    name = file.replace(".json", "")
                    config_files[name] = os.path.join(CONFIG_DIR, file)
        return config_files
    except Exception as e:
        logger.error(f"读取配置文件列表失败: {str(e)}")
        return {}

def create_default_config_if_missing():
    """如果默认配置文件不存在，则创建"""
    default_config_path = os.path.join(CONFIG_DIR, "advanced_training.json")
    if not os.path.exists(default_config_path):
        save_config(DEFAULT_TRAINING_PARAMS, default_config_path)
        logger.info(f"已创建默认高级训练配置文件: {default_config_path}")
    return default_config_path

def apply_preset(config, preset_name):
    """应用预设配置到现有配置"""
    if preset_name not in TRAINING_PRESETS:
        logger.warning(f"未找到预设配置: {preset_name}")
        return config
        
    preset = TRAINING_PRESETS[preset_name]
    
    # 深度递归合并预设配置
    def merge_dict(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                merge_dict(base[key], value)
            else:
                base[key] = value
    
    config_copy = config.copy()
    merge_dict(config_copy, preset)
    
    logger.info(f"已应用预设配置: {preset_name}")
    return config_copy

def get_sample_prompt_from_dataset(dataset_name):
    """从数据集名称生成示例提示词"""
    # 这里可以基于项目/数据集名称生成更智能的提示词
    return [dataset_name, f"{dataset_name} video"]

def update_video_dims_from_resolution(config, resolution):
    """从分辨率字符串更新配置中的视频尺寸"""
    try:
        parts = resolution.split('x')
        if len(parts) == 3:
            width, height, frames = map(int, parts)
            if "validation" in config:
                config["validation"]["video_dims"] = [width, height, frames]
            return True
    except:
        pass
    return False

def create_training_config_from_params(basename, model_path, resolution, rank, steps):
    """从基本参数创建训练配置"""
    config = DEFAULT_TRAINING_PARAMS.copy()
    
    # 更新基本信息
    config["model"]["model_source"] = str(model_path)
    config["lora"]["rank"] = rank
    config["lora"]["alpha"] = rank
    config["optimization"]["steps"] = steps
    config["validation"]["prompts"] = get_sample_prompt_from_dataset(basename)
    config["checkpoints"]["interval"] = steps
    config["validation"]["interval"] = steps
    
    # 设置输出目录
    output_dir = os.path.join(PROJECT_DIR, "outputs", f"{basename}_lora_training")
    config["output_dir"] = output_dir
    
    # 设置预处理数据根目录
    preprocessed_data_root = os.path.join(PROJECT_DIR, f"{basename}_scenes", ".precomputed")
    config["data"]["preprocessed_data_root"] = preprocessed_data_root
    
    # 从分辨率更新视频尺寸
    update_video_dims_from_resolution(config, resolution)
    
    return config
