import os
import sys
import yaml
from pathlib import Path

# 设置环境变量以处理编码问题
os.environ["PYTHONIOENCODING"] = "utf8"

# 强制离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 轻量配置
CONFIG = """
model:
  model_source: C:/COMFYUI/ComfyUI_windows_portable/ComfyUI/models/checkpoints/ltx-video-2b-v0.9.5.safetensors
  training_mode: lora
  load_checkpoint: null

lora:
  rank: 32
  alpha: 32
  dropout: 0.0
  target_modules:
    - to_k
    - to_q
    - to_v
    - to_out.0

optimization:
  learning_rate: 0.0002
  steps: 200
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
  compile_with_inductor: false
  compilation_mode: reduce-overhead

data:
  preprocessed_data_root: APT_scenes/.precomputed
  num_dataloader_workers: 0

validation:
  prompts:
    - "在视频中，一个女性角色正在跳舞"
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"
  images: null
  video_dims: [512, 512, 41]
  seed: 42
  inference_steps: 25
  interval: 50
  videos_per_prompt: 1
  guidance_scale: 3.5

checkpoints:
  interval: 50
  keep_last_n: -1

flow_matching:
  timestep_sampling_mode: shifted_logit_normal
  timestep_sampling_params: {}

seed: 42
output_dir: outputs/train_comfyui_model
"""

def main():
    print("\n=== ComfyUI模型训练 ===")
    print("使用本地ComfyUI模型文件直接训练")
    
    # 检查模型文件是否存在
    model_path = Path("C:/COMFYUI/ComfyUI_windows_portable/ComfyUI/models/checkpoints/ltx-video-2b-v0.9.5.safetensors")
    if not model_path.exists():
        print(f"错误: 未找到模型文件 {model_path}")
        return 1
    else:
        print(f"找到本地模型: {model_path}")
    
    # 创建配置并准备训练
    os.makedirs("configs", exist_ok=True)
    config_path = "configs/comfyui_train.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(CONFIG)
    
    print(f"配置已保存到: {config_path}")
    
    # 创建输出目录
    output_dir = "outputs/train_comfyui_model"
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        # 打印配置信息
        print("\n训练配置:")
        print("------------------------------------------------")
        print(f"模型源: {config_dict['model']['model_source']}")
        print(f"训练模式: {config_dict['model']['training_mode']}")
        print(f"LoRA 秩: {config_dict['lora']['rank']}")
        print(f"训练步数: {config_dict['optimization']['steps']}")
        print(f"批量大小: {config_dict['optimization']['batch_size']}")
        print(f"学习率: {config_dict['optimization']['learning_rate']}")
        print(f"混合精度模式: {config_dict['acceleration']['mixed_precision_mode']}")
        print(f"检查点保存间隔: {config_dict['checkpoints']['interval']}")
        print(f"输出目录: {config_dict['output_dir']}")
        print("------------------------------------------------")
        
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
