import os
import sys
import torch
from pathlib import Path

# 设置环境变量
os.environ["PYTHONIOENCODING"] = "utf8"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    print("\n=== 强制本地文件训练 ===")
    print("绕过HuggingFace下载，直接使用本地模型文件")
    
    # 指定ComfyUI中的模型文件路径
    comfyui_model_path = Path("C:/COMFYUI/ComfyUI_windows_portable/ComfyUI/models/checkpoints/ltx-video-2b-v0.9.5.safetensors")
    
    if not comfyui_model_path.exists():
        print(f"错误: 模型文件 {comfyui_model_path} 不存在")
        return 1
    
    print(f"找到本地模型: {comfyui_model_path}")
    print(f"模型文件大小: {comfyui_model_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    try:
        # 创建输出目录
        output_dir = "outputs/local_train"
        os.makedirs(output_dir, exist_ok=True)
        
        # 导入必要的模块
        from src.ltxv_trainer.config import LtxvTrainerConfig
        
        # 创建超轻量级配置
        config = {
            "model": {
                "model_source": str(comfyui_model_path),  # 直接使用本地文件路径
                "training_mode": "lora",
                "load_checkpoint": None
            },
            "lora": {
                "rank": 8,  # 更小的rank
                "alpha": 8,
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
                "steps": 25,  # 极少的步数
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
                "negative_prompt": "worst quality, inconsistent motion, blurry",
                "images": None,
                "video_dims": [128, 128, 8],  # 极小分辨率和帧数
                "seed": 42,
                "inference_steps": 10,  # 极少推理步数
                "interval": 25,
                "videos_per_prompt": 1,
                "guidance_scale": 3.5
            },
            "checkpoints": {
                "interval": 25,
                "keep_last_n": 1
            },
            "flow_matching": {
                "timestep_sampling_mode": "shifted_logit_normal",
                "timestep_sampling_params": {}
            },
            "seed": 42,
            "output_dir": output_dir
        }
        
        # 补丁替换模型加载函数
        def patch_model_loader():
            """直接替换核心加载函数，强制使用本地文件"""
            from diffusers import LTXVideoTransformer3DModel, AutoencoderKLLTXVideo
            
            # 保存原始方法
            original_from_single_file_auto = AutoencoderKLLTXVideo.from_single_file
            original_from_single_file_trans = LTXVideoTransformer3DModel.from_single_file
            
            # 替换为直接从本地文件加载的方法
            def force_local_load(cls, file_path, **kwargs):
                print(f"强制从本地加载: {file_path}")
                # 总是使用ComfyUI目录中的模型
                actual_path = str(comfyui_model_path)
                
                if 'AutoencoderKLLTXVideo' in str(cls):
                    return original_from_single_file_auto(cls, actual_path, **kwargs)
                else:
                    return original_from_single_file_trans(cls, actual_path, **kwargs)
            
            # 应用补丁
            AutoencoderKLLTXVideo.from_single_file = force_local_load
            LTXVideoTransformer3DModel.from_single_file = force_local_load
            
            print("已成功应用本地加载补丁")
        
        # 应用补丁
        patch_model_loader()
        
        # 打印配置信息
        print("\n训练配置:")
        print(f"- 本地模型路径: {config['model']['model_source']}")
        print(f"- 训练步数: {config['optimization']['steps']}")
        print(f"- 视频分辨率: {config['validation']['video_dims']}")
        print(f"- LoRA秩: {config['lora']['rank']}")
        
        # 导入trainer并训练
        from src.ltxv_trainer.trainer import LtxvTrainer
        trainer_config = LtxvTrainerConfig.parse_obj(config)
        
        print("\n正在初始化训练器...")
        trainer = LtxvTrainer(trainer_config)
        
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
