import os
import sys
import yaml
import torch
import json
import tempfile
from pathlib import Path
from collections import namedtuple

# 设置环境变量
os.environ["PYTHONIOENCODING"] = "utf8"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# VAE配置
VAE_CONFIG = {
    "_class_name": "AutoencoderKLLTXVideo",
    "_diffusers_version": "0.29.0.dev0",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": [
        "LTXVideoDownBlock3D",
        "LTXVideoDownBlock3D",
        "LTXVideoDownBlock3D",
        "LTXVideoDownBlock3D"
    ],
    "in_channels": 3,
    "latent_channels": 4,
    "layers_per_block": 2,
    "out_channels": 3,
    "up_block_types": [
        "LTXVideoUpBlock3D",
        "LTXVideoUpBlock3D",
        "LTXVideoUpBlock3D",
        "LTXVideoUpBlock3D"
    ]
}

# Transformer配置（2B模型）
TRANSFORMER_2B_CONFIG = {
    "_class_name": "LTXVideoTransformer3DModel",
    "_diffusers_version": "0.29.0.dev0",
    "activation_fn": "geglu",
    "attention_bias": False,
    "attention_head_dim": 64,
    "attention_out_bias": True,
    "caption_channels": 4096,
    "cross_attention_dim": 4096,
    "in_channels": 4,
    "norm_elementwise_affine": False,
    "norm_eps": 0.00005,
    "num_attention_heads": 16,
    "num_layers": 32,
    "out_channels": 4,
    "patch_size": 2,
    "patch_size_t": 1
}

def patch_transformers():
    """深度修补transformers库，使其不尝试下载任何东西"""
    from transformers import modeling_utils
    
    # 保存原始from_pretrained方法
    original_from_pretrained = modeling_utils.PreTrainedModel.from_pretrained
    
    # 替换方法
    def dummy_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        print(f"拦截模型加载请求: {pretrained_model_name_or_path}")
        
        # 创建一个默认实例
        model = cls(*model_args, **kwargs)
        print(f"已创建虚拟模型: {cls.__name__}")
        return model
    
    # 应用补丁
    modeling_utils.PreTrainedModel.from_pretrained = classmethod(dummy_from_pretrained)
    print("已替换PreTrainedModel.from_pretrained方法")
    
    # 同样替换tokenizer方法
    from transformers import tokenization_utils_base
    
    # 保存原始方法
    original_tokenizer_from_pretrained = tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained
    
    # 替换方法
    def dummy_tokenizer_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        print(f"拦截分词器加载请求: {pretrained_model_name_or_path}")
        
        # 创建一个简单的分词器
        if hasattr(cls, "__new__"):
            tokenizer = cls.__new__(cls)
            if hasattr(tokenizer, "__init__") and cls.__init__ is not object.__init__:
                # 尝试不带参数初始化
                try:
                    tokenizer.__init__()
                except:
                    # 如果失败，简单地pass
                    pass
        else:
            tokenizer = cls()
        
        # 添加必要的属性
        if not hasattr(tokenizer, "vocab"):
            tokenizer.vocab = {f"<token_{i}>": i for i in range(100)}
        if not hasattr(tokenizer, "model_max_length"):
            tokenizer.model_max_length = 512
        if not hasattr(tokenizer, "all_special_tokens"):
            tokenizer.all_special_tokens = ["<pad>", "</s>", "<unk>"]
        if not hasattr(tokenizer, "pad_token"):
            tokenizer.pad_token = "<pad>"
        if not hasattr(tokenizer, "eos_token"):
            tokenizer.eos_token = "</s>"
        if not hasattr(tokenizer, "unk_token"):
            tokenizer.unk_token = "<unk>"
        
        # 添加必要的方法
        if not hasattr(tokenizer, "__call__"):
            def tokenize_text(text, **kwargs):
                input_ids = [1] * 10  # 假设的token ids
                attention_mask = [1] * 10  # 全部注意力
                return {"input_ids": torch.tensor([input_ids]), "attention_mask": torch.tensor([attention_mask])}
            tokenizer.__call__ = tokenize_text
        
        print(f"已创建虚拟分词器: {cls.__name__}")
        return tokenizer
    
    # 应用补丁
    tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained = classmethod(dummy_tokenizer_from_pretrained)
    print("已替换PreTrainedTokenizerBase.from_pretrained方法")

def create_dummy_components(model_path):
    """创建虚拟的模型组件"""
    from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
    
    # 检查模型路径
    model_file = Path(model_path)
    if not model_file.exists():
        raise ValueError(f"模型文件不存在: {model_path}")
    
    print(f"加载本地模型文件: {model_path}")
    print(f"模型文件大小: {model_file.stat().st_size / (1024 * 1024):.1f} MB")
    
    # 确定使用哪个配置
    if "13b" in model_file.name.lower():
        transformer_config = TRANSFORMER_2B_CONFIG.copy()  # 调整为13B
        transformer_config["attention_head_dim"] = 128
        transformer_config["num_attention_heads"] = 32
        transformer_config["num_layers"] = 48
        print("检测到13B模型")
    else:
        transformer_config = TRANSFORMER_2B_CONFIG
        print("检测到2B模型")
    
    # 创建临时配置文件
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as vae_config_file:
        json.dump(VAE_CONFIG, vae_config_file)
        vae_config_path = vae_config_file.name
    
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as transformer_config_file:
        json.dump(transformer_config, transformer_config_file)
        transformer_config_path = transformer_config_file.name
    
    try:
        # 创建调度器
        scheduler = FlowMatchEulerDiscreteScheduler()
        
        # 创建分词器 (虚拟)
        class DummyTokenizer:
            def __init__(self):
                self.vocab = {f"<token_{i}>": i for i in range(100)}
                self.model_max_length = 512
                self.all_special_tokens = ["<pad>", "</s>", "<unk>"]
                self.pad_token = "<pad>"
                self.eos_token = "</s>"
                self.unk_token = "<unk>"
            
            def __call__(self, text, **kwargs):
                input_ids = [1] * 10  # 假设的token ids
                attention_mask = [1] * 10  # 全部注意力
                return {"input_ids": torch.tensor([input_ids]), "attention_mask": torch.tensor([attention_mask])}
        
        tokenizer = DummyTokenizer()
        
        # 创建文本编码器 (虚拟)
        class DummyTextEncoder:
            def __init__(self):
                self.config = type('obj', (object,), {'hidden_size': 4096})
            
            def __call__(self, input_ids=None, attention_mask=None, **kwargs):
                batch_size = input_ids.shape[0] if input_ids is not None else 1
                seq_len = input_ids.shape[1] if input_ids is not None else 10
                
                device = input_ids.device if input_ids is not None else "cpu"
                last_hidden_state = torch.randn(
                    (batch_size, seq_len, self.config.hidden_size), 
                    device=device,
                    dtype=torch.float16
                )
                
                return namedtuple('TextEncoderOutput', ['last_hidden_state'])(last_hidden_state=last_hidden_state)
            
            def to(self, device):
                return self
            
            def eval(self):
                return self
        
        text_encoder = DummyTextEncoder()
        
        # 加载真实的VAE和Transformer
        print("从本地文件加载VAE...")
        vae = AutoencoderKLLTXVideo.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            config=vae_config_path,
        )
        
        print("从本地文件加载Transformer...")
        transformer = LTXVideoTransformer3DModel.from_single_file(
            model_path, 
            torch_dtype=torch.float16,
            config=transformer_config_path,
        )
        
        # 创建组件集合
        Components = namedtuple('Components', [
            'scheduler', 'tokenizer', 'text_encoder', 'vae', 'transformer'
        ])
        
        components = Components(
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer
        )
        
        return components
    
    finally:
        # 清理临时文件
        try:
            Path(vae_config_path).unlink()
            Path(transformer_config_path).unlink()
        except:
            pass

def main():
    print("\n=== 完全本地训练模式 ===")
    print("使用全部本地化的组件，不尝试进行任何网络请求")
    
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
    
    # 优先使用2B模型
    model_path = None
    for model in model_files:
        if "2b" in model.name.lower():
            model_path = model
            break
    if model_path is None:
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
        "output_dir": "outputs/fully_local_train"
    }
    
    # 创建输出目录
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_path = "configs/fully_local_train.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"配置已保存到: {config_path}")
    
    # 修补transformers库
    patch_transformers()
    
    try:
        # 导入训练器和配置
        from src.ltxv_trainer.config import LtxvTrainerConfig
        from src.ltxv_trainer.trainer import LtxvTrainer
        
        # 创建训练器配置
        trainer_config = LtxvTrainerConfig.model_validate(config)
        
        # 打印配置信息
        print("\n训练配置:")
        print(f"- 本地模型路径: {config['model']['model_source']}")
        print(f"- 训练步数: {config['optimization']['steps']}")
        print(f"- 视频分辨率: {config['validation']['video_dims']}")
        print(f"- LoRA秩: {config['lora']['rank']}")
        print(f"- 输出目录: {config['output_dir']}")
        
        # 修补LtxvTrainer类
        def patch_trainer():
            # 保存原始方法
            original_load_models = LtxvTrainer._load_models
            
            # 替换方法
            def patched_load_models(self):
                print("使用完全本地化的组件加载...")
                model_path = self._config.model.model_source
                components = create_dummy_components(model_path)
                
                # 设置组件
                self._tokenizer = components.tokenizer
                self._text_encoder = components.text_encoder
                self._vae = components.vae
                self._transformer = components.transformer
                self._scheduler = components.scheduler
                
                print("所有组件已加载完成!")
            
            # 应用补丁
            LtxvTrainer._load_models = patched_load_models
            print("已修补LtxvTrainer._load_models方法")
        
        # 应用补丁
        patch_trainer()
        
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
