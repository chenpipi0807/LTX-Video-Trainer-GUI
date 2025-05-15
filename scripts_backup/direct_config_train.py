import os
import sys
from pathlib import Path

# 设置环境变量以处理编码问题
os.environ["PYTHONIOENCODING"] = "utf8"

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    print("\n=== 直接配置训练模式 ===")
    print("使用Python字典直接配置超轻量训练")
    
    try:
        # 导入必要模块
        from src.ltxv_trainer.model_loader import LtxvModelVersion
        from src.ltxv_trainer.config import LtxvTrainerConfig
        from src.ltxv_trainer.trainer import LtxvTrainer
        
        # 直接创建配置字典
        config = {
            "model": {
                "model_source": LtxvModelVersion.LTXV_2B_095,  # 使用枚举值
                "training_mode": "lora",
                "load_checkpoint": None
            },
            "lora": {
                "rank": 16,
                "alpha": 16,
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
                "steps": 50,
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
                "video_dims": [256, 256, 16],
                "seed": 42,
                "inference_steps": 20,
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
            "output_dir": "outputs/direct_config_train"
        }
        
        # 创建输出目录
        output_dir = "outputs/direct_config_train"
        os.makedirs(output_dir, exist_ok=True)
        
        # 打印关键配置
        print("\n训练配置:")
        print(f"- 模型源: {config['model']['model_source']}")
        print(f"- 训练步数: {config['optimization']['steps']}")
        print(f"- 视频分辨率: {config['validation']['video_dims']}")
        print(f"- LoRA秩: {config['lora']['rank']}")
        print(f"- 混合精度: {config['acceleration']['mixed_precision_mode']}")
        
        # 创建训练器配置
        trainer_config = LtxvTrainerConfig.parse_obj(config)
        
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
