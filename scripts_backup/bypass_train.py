import os
import sys
import yaml
from pathlib import Path

# 设置环境变量
os.environ["PYTHONIOENCODING"] = "utf8"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 首先修补模型加载器函数，然后再导入任何依赖库
def patch_loader_functions():
    """直接修补模型加载器函数以使用本地文件"""
    import torch
    from src.ltxv_trainer.model_loader import (
        load_vae,
        load_transformer,
        LtxvModelVersion,
        AutoencoderKLLTXVideo,
        LTXVideoTransformer3DModel,
    )
    
    # 定义新的加载函数
    def patched_load_vae(source, *, dtype=torch.bfloat16):
        print(f"拦截VAE加载请求，原始源: {source}")
        
        # 直接使用ComfyUI中的模型文件
        model_path = "C:/COMFYUI/ComfyUI_windows_portable/ComfyUI/models/checkpoints/ltx-video-2b-v0.9.5.safetensors"
        print(f"强制使用本地模型: {model_path}")
        
        return AutoencoderKLLTXVideo.from_single_file(model_path, torch_dtype=dtype)
    
    def patched_load_transformer(source, *, dtype=torch.float32):
        print(f"拦截Transformer加载请求，原始源: {source}")
        
        # 直接使用ComfyUI中的模型文件
        model_path = "C:/COMFYUI/ComfyUI_windows_portable/ComfyUI/models/checkpoints/ltx-video-2b-v0.9.5.safetensors"
        print(f"强制使用本地模型: {model_path}")
        
        return LTXVideoTransformer3DModel.from_single_file(model_path, torch_dtype=dtype)
    
    # 替换原始函数
    import src.ltxv_trainer.model_loader
    src.ltxv_trainer.model_loader.load_vae = patched_load_vae
    src.ltxv_trainer.model_loader.load_transformer = patched_load_transformer
    
    print("已成功修补模型加载函数")

# 应用修补
patch_loader_functions()

def main():
    print("\n=== 绕过下载训练 ===")
    print("完全绕过原始模型加载逻辑，强制使用本地文件")
    
    # 创建超轻量级训练配置
    config_dict = {
        "model": {
            "model_source": "LtxvModelVersion.LTXV_2B_095",  # 这个值其实已经不重要了
            "training_mode": "lora",
            "load_checkpoint": None
        },
        "lora": {
            "rank": 4,  # 极小LoRA秩
            "alpha": 4,
            "dropout": 0.0,
            "target_modules": [
                "to_k",
                "to_q",
                "to_v",
                "to_out.0"
            ]
        },
        "optimization": {
            "learning_rate": 0.0002,
            "steps": 10,  # 最少训练步数
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
            "prompts": ["在视频中，一个女性角色正在跳舞"],
            "negative_prompt": "worst quality",
            "images": None,
            "video_dims": [128, 128, 8],  # 极小分辨率
            "seed": 42,
            "inference_steps": 10,  # 最少推理步数
            "interval": 10,
            "videos_per_prompt": 1,
            "guidance_scale": 3.5
        },
        "checkpoints": {
            "interval": 10,
            "keep_last_n": 1
        },
        "flow_matching": {
            "timestep_sampling_mode": "shifted_logit_normal",
            "timestep_sampling_params": {}
        },
        "seed": 42,
        "output_dir": "outputs/bypass_train"
    }
    
    # 创建输出目录
    output_dir = "outputs/bypass_train"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 导入必要模块
        from src.ltxv_trainer.config import LtxvTrainerConfig
        from src.ltxv_trainer.trainer import LtxvTrainer
        
        # 创建训练配置
        trainer_config = LtxvTrainerConfig.parse_obj(config_dict)
        
        # 打印关键配置
        print("\n训练配置:")
        print(f"- 训练步数: {config_dict['optimization']['steps']}")
        print(f"- 视频分辨率: {config_dict['validation']['video_dims']}")
        print(f"- LoRA秩: {config_dict['lora']['rank']}")
        print(f"- 混合精度: {config_dict['acceleration']['mixed_precision_mode']}")
        
        # 初始化训练器
        print("\n正在初始化训练器...")
        trainer = LtxvTrainer(trainer_config)
        
        # 开始训练
        print("\n开始训练...")
        result_path, stats = trainer.train()
        
        print(f"\n训练完成！")
        print(f"最终检查点保存在: {result_path}")
        print(f"训练用时: {stats.total_time_seconds / 60:.1f} 分钟")
        
    except Exception as e:
        import traceback
        print(f"\n训练过程中出错:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
