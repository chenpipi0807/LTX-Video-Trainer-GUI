import os
import sys
import yaml
from pathlib import Path

# 强制离线模式和环境设置
os.environ["PYTHONIOENCODING"] = "utf8"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 指定ComfyUI模型路径作为预训练权重路径
os.environ["PRETRAINED_MODEL_DIR"] = str(Path("C:/COMFYUI/ComfyUI_windows_portable/ComfyUI/models/checkpoints"))

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 轻量配置
CONFIG = """
model:
  model_source: LTXV_2B_095  # 使用较小的2B模型, 这个模型ComfyUI中有
  training_mode: lora
  load_checkpoint: null

lora:
  rank: 32  # 更轻量的LoRA
  alpha: 32
  dropout: 0.0
  target_modules:
    - to_k
    - to_q
    - to_v
    - to_out.0

optimization:
  learning_rate: 0.0002
  steps: 200  # 更少的步数
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
  compile_with_inductor: false  # 禁用编译
  compilation_mode: reduce-overhead

data:
  preprocessed_data_root: APT_scenes/.precomputed
  num_dataloader_workers: 0

validation:
  prompts:
    - "在视频中，一个女性角色正在跳舞"
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"
  images: null
  video_dims: [512, 512, 41]  # 较小尺寸
  seed: 42
  inference_steps: 25  # 较少推理步数
  interval: 50  # 更频繁验证
  videos_per_prompt: 1
  guidance_scale: 3.5

checkpoints:
  interval: 50  # 更频繁保存
  keep_last_n: -1

flow_matching:
  timestep_sampling_mode: shifted_logit_normal
  timestep_sampling_params: {}

seed: 42
output_dir: outputs/train_comfyui_model
"""

def monkey_patch_model_loader():
    """猴子补丁替换模型加载过程，强制使用本地模型"""
    from src.ltxv_trainer.model_loader import MODEL_SOURCES
    
    # 添加ComfyUI模型路径前缀
    comfyui_dir = os.environ.get("PRETRAINED_MODEL_DIR")
    
    # 重新映射模型源到本地文件路径
    MODEL_SOURCES["LTXV_2B_095"] = f"{comfyui_dir}/ltx-video-2b-v0.9.5.safetensors"
    MODEL_SOURCES["LTXV_13B_097_DEV"] = f"{comfyui_dir}/ltxv-13b-0.9.7-dev.safetensors"
    
    # 打印映射信息
    print(f"\n模型源已映射到ComfyUI目录:")
    for k, v in MODEL_SOURCES.items():
        print(f"  {k} -> {v}")

def main():
    print("\n=== ComfyUI模型训练 ===")
    print(f"使用目录: {os.environ.get('PRETRAINED_MODEL_DIR')}")
    
    # 创建配置并准备训练
    os.makedirs("configs", exist_ok=True)
    config_path = "configs/train_comfyui.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(CONFIG)
    
    print(f"配置已保存到: {config_path}")
    
    # 创建输出目录
    output_dir = "outputs/train_comfyui_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # 应用猴子补丁修改模型加载逻辑
    monkey_patch_model_loader()
    
    try:
        # 加载配置
        print("加载训练配置...")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        # 动态导入所需模块
        from src.ltxv_trainer.config import LtxvTrainerConfig
        from src.ltxv_trainer.trainer import LtxvTrainer
        
        # 创建训练配置
        trainer_config = LtxvTrainerConfig(**config_dict)
        print("配置加载成功，开始训练...")
        
        # 打印配置
        for key, value in config_dict.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        # 初始化trainer并训练
        print("\n正在初始化训练器...")
        trainer = LtxvTrainer(trainer_config)
        
        print("\n开始训练...")
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
