#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
直接训练脚本，避免编码问题
"""

import os
import sys
import yaml
import shutil
from pathlib import Path

# 设置环境变量以处理编码问题
os.environ["PYTHONIOENCODING"] = "utf8"

# 设置使用本地模型而不是下载
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 动态导入所需模块
from src.ltxv_trainer.config import LtxvTrainerConfig
from src.ltxv_trainer.trainer import LtxvTrainer

# 轻量级训练配置
CONFIG = """
model:
  model_source: LTXV_13B_097_DEV
  training_mode: lora
  load_checkpoint: null

lora:
  rank: 64  # 降低了LoRA秩，减少内存使用
  alpha: 64
  dropout: 0.0
  target_modules:
    - to_k
    - to_q
    - to_v
    - to_out.0

optimization:
  learning_rate: 0.0002
  steps: 300  # 减少训练步数
  batch_size: 1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  optimizer_type: adamw
  scheduler_type: linear
  scheduler_params: {}
  enable_gradient_checkpointing: true
  first_frame_conditioning_p: 0.5

acceleration:
  mixed_precision_mode: bf16
  quantization: null
  load_text_encoder_in_8bit: false
  compile_with_inductor: false  # 禁用编译以避免潜在的问题
  compilation_mode: reduce-overhead

data:
  preprocessed_data_root: APT_scenes/.precomputed
  num_dataloader_workers: 0  # 保持0以避免序列化问题

validation:
  prompts:
    - "在视频中，一个女性角色正在跳舞"
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"
  images: null
  video_dims: [512, 512, 41]  # 使用更小的尺寸和更短的视频长度
  seed: 42
  inference_steps: 25  # 减少推理步数以加快验证
  interval: 100  # 更频繁地创建验证视频
  videos_per_prompt: 1
  guidance_scale: 3.5

checkpoints:
  interval: 100  # 更频繁地保存检查点
  keep_last_n: -1

flow_matching:
  timestep_sampling_mode: shifted_logit_normal
  timestep_sampling_params: {}

seed: 42
output_dir: outputs/train_lite_output
"""

def setup_model_paths():
    """设置模型路径，确保可以在离线模式下找到模型"""
    # 检查ComfyUI目录
    comfyui_model_path = Path("C:/COMFYUI/ComfyUI_windows_portable/ComfyUI/models/checkpoints")
    target_model_name = "ltxv-13b-0.9.7-dev.safetensors"
    
    if comfyui_model_path.exists() and (comfyui_model_path / target_model_name).exists():
        print(f"\n发现本地模型: {comfyui_model_path / target_model_name}")
        
        # 创建Transformers缓存目录
        cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
        model_dir = cache_dir / "models--Lightricks--LTXV_13B_097_DEV" / "snapshots" / "refs/tags/v1.0.0"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建指向ComfyUI模型的符号链接
        model_path = model_dir / target_model_name
        if not model_path.exists():
            print(f"创建指向ComfyUI模型的软链接...")
            try:
                # 复制而不是创建符号链接，因为Windows权限问题
                shutil.copy2(comfyui_model_path / target_model_name, model_path)
                print(f"已复制模型到Transformers缓存路径: {model_path}")
            except Exception as e:
                print(f"复制模型文件时出错: {e}")
                return False
        else:
            print(f"模型已存在于缓存目录: {model_path}")
        
        # 创建配置文件
        config_file = model_dir / "config.json"
        if not config_file.exists():
            # 写入简化的配置
            with open(config_file, "w", encoding="utf-8") as f:
                f.write('{"_name_or_path": "Lightricks/LTXV_13B_097_DEV"}')
            print("已创建模型配置文件")
        
        return True
    else:
        print(f"\n警告: 未找到ComfyUI中的模型 {target_model_name}")
        print(f"尝试寻找的路径: {comfyui_model_path}")
        print("即将尝试从网络下载模型...")
        return False

def main():
    print("=== 开始直接训练 ===")
    print("这将使用轻量级配置进行训练，以避免Windows编码问题")
    
    # 设置模型路径
    setup_model_paths()
    
    # 创建输出目录
    output_dir = "outputs/train_lite_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建配置目录
    os.makedirs("configs", exist_ok=True)
    
    # 写入轻量级配置
    config_path = "configs/direct_train_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(CONFIG)
    
    print(f"创建了轻量级训练配置: {config_path}")
    
    # 确保预处理数据目录存在
    if not os.path.exists("APT_scenes/.precomputed"):
        print("错误: 预处理数据目录不存在。请先运行预处理步骤。")
        return
    
    try:
        # 加载配置
        print("加载训练配置...")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        # 创建训练配置
        trainer_config = LtxvTrainerConfig(**config_dict)
        print("配置加载成功")
        
        # 使用拍平版本的配置打印，避免rich库的编码问题
        print("\n=== 训练配置 ===")
        for key, value in config_dict.items():
            print(f"{key}: {value}")
        
        # 初始化trainer并训练
        print("\n开始训练...")
        trainer = LtxvTrainer(trainer_config)
        result_path, stats = trainer.train(disable_progress_bars=False)
        
        print(f"\n训练完成！")
        print(f"最终检查点保存在: {result_path}")
        print(f"训练用时: {stats.total_time_seconds / 60:.1f} 分钟")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
