import os
import sys
import yaml
import torch
import traceback
from pathlib import Path

# 设置环境变量
os.environ["PYTHONIOENCODING"] = "utf8"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    print("\n=== 修复版本训练 ===")
    print("绕过原始模型加载器实现，直接使用本地文件")
    
    # 检查模型文件存在
    comfyui_model_path = Path("C:/COMFYUI/ComfyUI_windows_portable/ComfyUI/models/checkpoints/ltx-video-2b-v0.9.5.safetensors")
    if not comfyui_model_path.exists():
        print(f"错误: 模型文件不存在: {comfyui_model_path}")
        return 1
    
    print(f"找到本地模型: {comfyui_model_path}")
    print(f"模型文件大小: {comfyui_model_path.stat().st_size / (1024 * 1024):.1f} MB")
    
    # 创建一个配置文件
    config_path = "configs/fix_run_train.yaml"
    
    # 定义一个最小的训练配置
    config = {
        "model": {
            "model_source": str(comfyui_model_path),  # 直接使用文件路径
            "training_mode": "lora",
            "load_checkpoint": None
        },
        "lora": {
            "rank": 4,  # 极小的rank
            "alpha": 4,
            "dropout": 0.0,
            "target_modules": [
                "to_k", "to_q", "to_v", "to_out.0"
            ]
        },
        "optimization": {
            "learning_rate": 0.0002,
            "steps": 5,  # 最少的步数
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
            "preprocessed_data_root": "APT_scenes/.precomputed",
            "num_dataloader_workers": 0
        },
        "validation": {
            "prompts": ["简单的舞蹈视频"],
            "negative_prompt": "worst quality",
            "images": None,
            "video_dims": [128, 128, 8],  # 极小分辨率
            "seed": 42,
            "inference_steps": 5,  # 最少推理步数
            "interval": 5,
            "videos_per_prompt": 1,
            "guidance_scale": 3.5
        },
        "checkpoints": {
            "interval": 5,
            "keep_last_n": 1
        },
        "flow_matching": {
            "timestep_sampling_mode": "shifted_logit_normal",
            "timestep_sampling_params": {}
        },
        "seed": 42,
        "output_dir": "outputs/fix_run_train"
    }
    
    # 保存配置
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"已保存配置到: {config_path}")
    
    # 修补关键函数
    def monkey_patch_system():
        print("正在修补系统函数...")
        
        # 重要：修补_is_safetensors_url函数，使其能识别本地文件
        from src.ltxv_trainer.model_loader import _is_safetensors_url
        
        def patched_is_safetensors_url(source):
            """修改后的函数可以识别本地.safetensors文件"""
            source_str = str(source)
            return source_str.endswith('.safetensors')
        
        # 应用补丁
        import src.ltxv_trainer.model_loader
        src.ltxv_trainer.model_loader._is_safetensors_url = patched_is_safetensors_url
        
        # 修补from_single_file方法，打印更多信息
        from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
        
        # 保存原始方法
        orig_vae_from_single_file = AutoencoderKLLTXVideo.from_single_file
        orig_transformer_from_single_file = LTXVideoTransformer3DModel.from_single_file
        
        # 创建新方法
        def verbose_vae_from_single_file(cls, file_path, **kwargs):
            print(f"从文件加载VAE: {file_path}")
            print(f"参数: {kwargs}")
            return orig_vae_from_single_file(cls, file_path, **kwargs)
        
        def verbose_transformer_from_single_file(cls, file_path, **kwargs):
            print(f"从文件加载Transformer: {file_path}")
            print(f"参数: {kwargs}")
            return orig_transformer_from_single_file(cls, file_path, **kwargs)
        
        # 应用补丁
        AutoencoderKLLTXVideo.from_single_file = classmethod(verbose_vae_from_single_file)
        LTXVideoTransformer3DModel.from_single_file = classmethod(verbose_transformer_from_single_file)
        
        print("系统函数修补完成")
    
    # 应用修补
    monkey_patch_system()
    
    try:
        # 创建输出目录
        output_dir = config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建训练器配置
        from src.ltxv_trainer.config import LtxvTrainerConfig
        trainer_config = LtxvTrainerConfig.parse_obj(config)
        
        # 打印配置信息
        print("\n训练配置:")
        print(f"- 模型路径: {config['model']['model_source']}")
        print(f"- 训练步数: {config['optimization']['steps']}")
        print(f"- 视频分辨率: {config['validation']['video_dims']}")
        print(f"- LoRA秩: {config['lora']['rank']}")
        print(f"- 输出目录: {config['output_dir']}")
        
        # 初始化训练器
        from src.ltxv_trainer.trainer import LtxvTrainer
        print("\n初始化训练器...")
        trainer = LtxvTrainer(trainer_config)
        
        # 开始训练
        print("\n开始训练...")
        result_path, stats = trainer.train()
        
        print(f"\n训练完成!")
        print(f"模型保存在: {result_path}")
        print(f"训练用时: {stats.total_time_seconds / 60:.1f} 分钟")
        
        return 0
        
    except Exception as e:
        print("\n训练失败:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
