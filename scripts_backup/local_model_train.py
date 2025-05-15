import os
import sys
import yaml
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
    print("\n=== 本地模型训练 ===")
    print("使用完全本地化的模型加载器，直接从models目录加载模型")
    
    # 确保models目录存在
    models_dir = project_root / "models"
    if not models_dir.exists():
        os.makedirs(models_dir, exist_ok=True)
        print(f"已创建models目录: {models_dir}")
    
    # 检查models目录中的模型文件
    model_files = list(models_dir.glob("*.safetensors"))
    if not model_files:
        print("警告: models目录中未找到.safetensors模型文件")
        print(f"请将您的模型文件复制到: {models_dir}")
        return 1
    
    print(f"发现本地模型文件:")
    for i, model_file in enumerate(model_files):
        print(f"  {i+1}. {model_file.name} ({model_file.stat().st_size / (1024 * 1024):.1f} MB)")
    
    # 使用第一个发现的模型文件
    model_path = model_files[0]
    print(f"\n将使用模型: {model_path}")
    
    # 创建超轻量级训练配置
    config = {
        "model": {
            "model_source": str(model_path),  # 直接使用本地文件路径
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
            "steps": 10,  # 最少步数
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
        "output_dir": "outputs/local_model_train"
    }
    
    # 创建输出目录
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_path = "configs/local_model_train.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"配置已保存到: {config_path}")
    
    try:
        # 导入我们的本地模型加载器
        from src.ltxv_trainer.local_model_loader import load_local_components
        
        # 修补LtxvTrainer类以使用我们的本地加载器
        def patch_trainer_class():
            from src.ltxv_trainer.trainer import LtxvTrainer
            
            # 保存原始_load_models方法
            original_load_models = LtxvTrainer._load_models
            
            # 创建新的替代方法
            def patched_load_models(self):
                """替换为使用本地模型加载器的方法"""
                print("使用本地模型加载器...")
                
                # 获取配置中的模型路径
                model_path = self._config.model.model_source
                
                # 使用我们的本地加载器
                # 转换混合精度模式为PyTorch数据类型
                precision_mode = self._config.acceleration.mixed_precision_mode
                if precision_mode == "fp16":
                    dtype = torch.float16
                elif precision_mode == "bf16":
                    dtype = torch.bfloat16
                else:  # "no"
                    dtype = torch.float32
                    
                components = load_local_components(
                    model_path=model_path,
                    load_text_encoder_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
                    transformer_dtype=dtype,
                )
                
                # 设置组件
                self._tokenizer = components.tokenizer
                self._text_encoder = components.text_encoder
                self._vae = components.vae
                self._transformer = components.transformer
                self._scheduler = components.scheduler
                
                print("所有模型组件已成功加载")
            
            # 应用补丁
            import torch
            LtxvTrainer._load_models = patched_load_models
            print("已成功修补LtxvTrainer._load_models方法")
        
        # 应用修补
        patch_trainer_class()
        
        # 导入训练器和配置
        from src.ltxv_trainer.config import LtxvTrainerConfig
        from src.ltxv_trainer.trainer import LtxvTrainer
        
        # 创建训练器配置
        trainer_config = LtxvTrainerConfig.parse_obj(config)
        
        # 打印配置信息
        print("\n训练配置:")
        print(f"- 本地模型路径: {config['model']['model_source']}")
        print(f"- 训练步数: {config['optimization']['steps']}")
        print(f"- 视频分辨率: {config['validation']['video_dims']}")
        print(f"- LoRA秩: {config['lora']['rank']}")
        print(f"- 输出目录: {config['output_dir']}")
        
        # 初始化训练器
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
        import traceback
        print("\n训练失败:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
