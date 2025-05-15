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

def main():
    print("\n=== 极简训练模式 ===")
    print("使用最小化配置进行快速训练")
    
    # 检查数据目录
    data_dir = Path("APT_scenes/.precomputed")
    if not data_dir.exists():
        print(f"警告: 预处理数据目录不存在: {data_dir}")
        print("您可能需要先运行预处理步骤")
    
    # 创建训练配置
    config_path = "configs/minimal_train.yaml"
    config = {
        "model": {
            "model_source": "Lightricks/LTX-Video-0.9.5",  # 使用官方源，但固定为2B模型
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
            "steps": 5,  # 最少步数
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
        "output_dir": "outputs/minimal_train"
    }
    
    # 保存配置
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"已保存配置到: {config_path}")
    print("\n我们将使用在线模式进行一次最小化训练")
    print("目标是验证训练能够正常完成，生成最基本的LoRA模型")
    print("训练将只持续5步，分辨率降至最低，以确保能快速完成")
    
    # 创建输出目录
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 导入必要模块
        from src.ltxv_trainer.config import LtxvTrainerConfig
        from src.ltxv_trainer.trainer import LtxvTrainer
        
        # 创建训练器配置
        trainer_config = LtxvTrainerConfig.parse_obj(config)
        
        # 打印配置信息
        print("\n训练配置:")
        print(f"- 模型源: {config['model']['model_source']}")
        print(f"- 训练步数: {config['optimization']['steps']}")
        print(f"- 视频分辨率: {config['validation']['video_dims']}")
        print(f"- LoRA秩: {config['lora']['rank']}")
        print(f"- 输出目录: {config['output_dir']}")
        
        # 初始化训练器
        print("\n初始化训练器...")
        # 暂时取消offline模式以允许下载模型
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "0"
        
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
