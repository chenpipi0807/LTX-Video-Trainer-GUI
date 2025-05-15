import json
import tempfile
from enum import Enum
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import torch
from diffusers import (
    AutoencoderKLLTXVideo,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    LTXVideoTransformer3DModel,
)
from pydantic import BaseModel, ConfigDict
from transformers import T5EncoderModel, T5Tokenizer

# The main HF repo to load scheduler, tokenizer, and text encoder from
HF_MAIN_REPO = "Lightricks/LTX-Video"

# Default local diffusers model path
DEFAULT_DIFFUSERS_PATH = Path("models/LTX-Video-0.9.7-diffusers")


class LtxvModelVersion(str, Enum):
    """Available LTXV model versions."""

    LTXV_2B_090 = "LTXV_2B_0.9.0"
    LTXV_2B_091 = "LTXV_2B_0.9.1"
    LTXV_2B_095 = "LTXV_2B_0.9.5"
    LTXV_13B_097_DEV = "LTXV_13B_097_DEV"

    def __str__(self) -> str:
        """Return the version string."""
        return self.value

    @classmethod
    def latest(cls) -> "LtxvModelVersion":
        """Get the latest available version."""
        return cls.LTXV_2B_095

    @property
    def hf_repo(self) -> str:
        """Get the HuggingFace repo for this version."""
        match self:
            case LtxvModelVersion.LTXV_2B_090:
                return "Lightricks/LTX-Video"
            case LtxvModelVersion.LTXV_2B_091:
                return "Lightricks/LTX-Video-0.9.1"
            case LtxvModelVersion.LTXV_2B_095:
                return "Lightricks/LTX-Video-0.9.5"
            case LtxvModelVersion.LTXV_13B_097_DEV:
                raise ValueError("LTXV_13B_097_DEV does not have a HuggingFace repo")
        raise ValueError(f"Unknown version: {self}")

    @property
    def safetensors_url(self) -> str:
        """Get the safetensors URL for this version."""
        match self:
            case LtxvModelVersion.LTXV_2B_090:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.safetensors"
            case LtxvModelVersion.LTXV_2B_091:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors"
            case LtxvModelVersion.LTXV_2B_095:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.5.safetensors"
            case LtxvModelVersion.LTXV_13B_097_DEV:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-dev.safetensors"
        raise ValueError(f"Unknown version: {self}")


# Type for model sources - can be:
# 1. HuggingFace repo ID (str)
# 2. Local path (str or Path)
# 3. Direct version specification (LtxvModelVersion)
ModelSource = Union[str, Path, LtxvModelVersion]


class LtxvModelComponents(BaseModel):
    """Container for all LTXV model components."""

    scheduler: FlowMatchEulerDiscreteScheduler
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel
    vae: AutoencoderKLLTXVideo
    transformer: LTXVideoTransformer3DModel

    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_scheduler() -> FlowMatchEulerDiscreteScheduler:
    """
    Load the Flow Matching scheduler component from the main HF repo.

    Returns:
        Loaded scheduler
    """
    # First try to load from local diffusers path
    scheduler_path = DEFAULT_DIFFUSERS_PATH / "scheduler"
    if scheduler_path.exists():
        try:
            print(f"从本地diffusers目录加载scheduler: {scheduler_path}")
            return FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(scheduler_path),
                local_files_only=True,
            )
        except Exception as e:
            print(f"从本地加载scheduler失败: {e}，尝试从主仓库加载")
    
    # Fall back to online loading
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        HF_MAIN_REPO,
        subfolder="scheduler",
    )


def load_tokenizer() -> T5Tokenizer:
    """
    Load the T5 tokenizer component from the main HF repo.

    Returns:
        Loaded tokenizer
    """
    # First try to load from local diffusers path
    tokenizer_path = DEFAULT_DIFFUSERS_PATH / "tokenizer"
    if tokenizer_path.exists():
        try:
            print(f"从本地diffusers目录加载tokenizer: {tokenizer_path}")
            return T5Tokenizer.from_pretrained(
                str(tokenizer_path),
                local_files_only=True,
            )
        except Exception as e:
            print(f"从本地加载tokenizer失败: {e}，尝试从主仓库加载")
    
    # Try to load from models/t5-base if it exists
    t5_tokenizer_path = Path("models/t5-base")
    if t5_tokenizer_path.exists():
        try:
            print(f"从本地T5模型目录加载tokenizer: {t5_tokenizer_path}")
            return T5Tokenizer.from_pretrained(
                str(t5_tokenizer_path),
                local_files_only=True,
            )
        except Exception as e:
            print(f"从本地T5模型目录加载tokenizer失败: {e}")
            # 在失败的情况下尝试使用虚拟分词器
            try:
                from .local_model_loader import VirtualT5Tokenizer
                print("使用虚拟T5分词器进行离线操作")
                return VirtualT5Tokenizer()
            except ImportError:
                print("未找到local_model_loader模块，无法创建虚拟分词器")
    
    # Fall back to online loading
    return T5Tokenizer.from_pretrained(
        HF_MAIN_REPO,
        subfolder="tokenizer",
    )


def load_text_encoder(*, load_in_8bit: bool = False) -> T5EncoderModel:
    """
    Load the T5 text encoder component from the main HF repo.

    Args:
        load_in_8bit: Whether to load in 8-bit precision

    Returns:
        Loaded text encoder
    """
    kwargs = (
        {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        if load_in_8bit
        else {"torch_dtype": torch.bfloat16}
    )
    
    # First try to load from local diffusers path
    text_encoder_path = DEFAULT_DIFFUSERS_PATH / "text_encoder"
    if text_encoder_path.exists():
        try:
            print(f"从本地diffusers目录加载text_encoder: {text_encoder_path}")
            return T5EncoderModel.from_pretrained(
                str(text_encoder_path),
                local_files_only=True,
                **kwargs
            )
        except Exception as e:
            print(f"从本地加载text_encoder失败: {e}，尝试从主仓库加载")
    
    # Try to load from models/t5-base if it exists
    t5_encoder_path = Path("models/t5-base")
    if t5_encoder_path.exists():
        try:
            print(f"从本地T5模型目录加载text_encoder: {t5_encoder_path}")
            return T5EncoderModel.from_pretrained(
                str(t5_encoder_path),
                local_files_only=True,
                **kwargs
            )
        except Exception as e:
            print(f"从本地T5模型目录加载text_encoder失败: {e}")
            # 在失败的情况下尝试使用虚拟模型
            try:
                from .local_model_loader import VirtualT5Encoder
                print("使用虚拟T5编码器进行离线操作")
                return VirtualT5Encoder()
            except ImportError:
                print("未找到local_model_loader模块，无法创建虚拟编码器")
    
    # Fall back to online loading
    return T5EncoderModel.from_pretrained(HF_MAIN_REPO, subfolder="text_encoder", **kwargs)


def load_vae(
    source: ModelSource,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> AutoencoderKLLTXVideo:
    """
    Load the VAE component.

    Args:
        source: Model source (HF repo, local path, or version)
        dtype: Data type for the VAE
        is_offline: Whether to use virtual VAE in offline mode

    Returns:
        Loaded VAE
    """
    import os

    # 尝试加载真实VAE并处理兼容性问题
    def load_vae_with_compatibility_fix(vae_path):
        """加载VAE并处理兼容性问题"""
        import json
        import os
        
        config_path = os.path.join(vae_path, "config.json")
        weights_path = os.path.join(vae_path, "diffusion_pytorch_model.safetensors")
        
        if not (os.path.exists(config_path) and os.path.exists(weights_path)):
            print(f"缺少必要的VAE文件: {vae_path}")
            return None
            
        try:
            # 加载和修改配置文件
            with open(config_path, "r") as f:
                config = json.load(f)
                
            # 检查并处理可能存在兼容性问题的字段
            layers_per_block = config.get("layers_per_block", [])
            block_out_channels = config.get("block_out_channels", [])
            spatio_temporal_scaling = config.get("spatio_temporal_scaling", [])
            down_block_types = config.get("down_block_types", [])
            
            # 确保 spatio_temporal_scaling 长度与 block_out_channels 一致
            if len(spatio_temporal_scaling) < len(block_out_channels):
                print(f"修复兼容性问题: spatio_temporal_scaling 长度({len(spatio_temporal_scaling)}) < block_out_channels 长度({len(block_out_channels)})")
                # 扩展 spatio_temporal_scaling 数组到必要的长度
                while len(spatio_temporal_scaling) < len(block_out_channels):
                    spatio_temporal_scaling.append(True)  # 使用True作为默认值
                config["spatio_temporal_scaling"] = spatio_temporal_scaling
                print(f"新的spatio_temporal_scaling: {spatio_temporal_scaling}")
                
            # 修复其他可能不兼容的长度字段
            decoder_keys = [k for k in config.keys() if k.startswith("decoder_") and isinstance(config[k], list)]
            for key in decoder_keys:
                if len(config[key]) != len(config.get("decoder_block_out_channels", [])):
                    print(f"修复兼容性问题: {key} 长度不匹配")
                    # 调整长度便于匹配
                    target_len = len(config.get("decoder_block_out_channels", []))
                    if len(config[key]) < target_len:
                        # 复制最后一个元素
                        last_val = config[key][-1] if config[key] else True  # 默认值
                        while len(config[key]) < target_len:
                            config[key].append(last_val)
                    elif len(config[key]) > target_len:
                        # 截断到目标长度
                        config[key] = config[key][:target_len]
            
            # 将修改后的配置写入临时文件
            import tempfile
            temp_config = tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False)
            json.dump(config, temp_config)
            temp_config.flush()
            temp_config_path = temp_config.name
            temp_config.close()
            
            # 使用修改后的配置文件加载模型
            print(f"使用修改后的配置加载 VAE: {vae_path}")
            model = AutoencoderKLLTXVideo.from_pretrained(
                pretrained_model_name_or_path=os.path.dirname(vae_path),
                subfolder=os.path.basename(vae_path),
                torch_dtype=dtype,
                local_files_only=True,
                config_file=temp_config_path,
                low_cpu_mem_usage=False,
                device_map=None
            )
            
            # 清理临时文件
            try:
                os.unlink(temp_config_path)
            except:
                pass
                
            return model
        except Exception as e:
            print(f"使用兼容性修复加载VAE失败: {e}")
            # 加载真实VAE失败，返回None
            return None
    
    # 尝试带兼容性修复的方式加载VAE
    diffusers_path = DEFAULT_DIFFUSERS_PATH
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if source_path.exists() and source_path.is_dir():
            # 检查是否为diffusers格式目录
            if (source_path / "vae").exists() and (source_path / "model_index.json").exists():
                diffusers_path = source_path

    # 尝试从diffusers目录加载
    vae_path = diffusers_path / "vae"
    if vae_path.exists():
        print(f"使用兼容性模式从本地diffusers目录加载VAE: {vae_path}")
        vae = load_vae_with_compatibility_fix(str(vae_path))
        if vae is not None:
            return vae
        print("兼容性加载失败，尝试其他加载方式...")
    
    # 直接尝试加载VAE
    
    # 当前代码已经在上面尝试加载diffusers目录，这里不再重复

    # Try to load from a version enum
    if isinstance(source, LtxvModelVersion):
        try:
            return AutoencoderKLLTXVideo.from_pretrained(
                source.hf_repo,
                subfolder="vae",
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
                device_map=None
            )
        except ValueError:
            # This version doesn't have a HF repo, try safetensors URL
            return AutoencoderKLLTXVideo.from_single_file(
                source.safetensors_url,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
                device_map=None
            )

    # Try to load from a string (HF repo, local path, or version name)
    if isinstance(source, (str, Path)):
        source_str = str(source)

        # Try to parse as version
        version = _try_parse_version(source_str)
        if version:
            return load_vae(version, dtype=dtype)

        # Try to load directly from HF repo
        if _is_huggingface_repo(source_str):
            return AutoencoderKLLTXVideo.from_pretrained(
                source_str,
                subfolder="vae",
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
                device_map=None
            )

        # Try to load from local path
        if Path(source_str).exists():
            # Check if it's a directory with a "vae" subfolder
            if (Path(source_str) / "vae").exists():
                return AutoencoderKLLTXVideo.from_pretrained(
                    source_str,
                    subfolder="vae",
                    torch_dtype=dtype,
                    local_files_only=True,
                    low_cpu_mem_usage=False,
                    device_map=None
                )

            # Check if it's a single .safetensors file
            if source_str.endswith(".safetensors") and Path(source_str).is_file():
                return AutoencoderKLLTXVideo.from_single_file(
                    source_str,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=False,
                    device_map=None
                )
    
    raise ValueError(f"Invalid model source: {source}")


def load_transformer(
    source: ModelSource,
    *,
    dtype: torch.dtype = torch.float32,
    is_offline: bool = False,
) -> LTXVideoTransformer3DModel:
    """
    Load the transformer component.

    Args:
        source: Model source (HF repo, local path, or version)
        dtype: Data type for the transformer

    Returns:
        Loaded transformer
    """
    # Try to load from diffusers format directory
    diffusers_path = DEFAULT_DIFFUSERS_PATH
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if source_path.exists() and source_path.is_dir():
            # Check if this is a diffusers directory
            if (source_path / "transformer").exists() and (source_path / "model_index.json").exists():
                diffusers_path = source_path
    
    # Try loading from the diffusers directory first
    transformer_path = diffusers_path / "transformer"
    if transformer_path.exists():
        try:
            print(f"从本地diffusers目录加载transformer: {transformer_path}")
            return LTXVideoTransformer3DModel.from_pretrained(
                str(transformer_path),
                torch_dtype=dtype,
                local_files_only=True,
            )
        except Exception as e:
            print(f"从diffusers目录加载transformer失败: {e}，尝试其他加载方法")
    
    # Try to load from a version enum
    if isinstance(source, LtxvModelVersion):
        try:
            if source == LtxvModelVersion.LTXV_13B_097_DEV:
                # Special case for 13B model
                return _load_ltxv_13b_transformer(source.safetensors_url, dtype=dtype)

            return LTXVideoTransformer3DModel.from_pretrained(
                source.hf_repo,
                subfolder="transformer",
                torch_dtype=dtype,
            )
        except ValueError:
            # This version doesn't have a HF repo, try safetensors URL
            return LTXVideoTransformer3DModel.from_single_file(
                source.safetensors_url,
                torch_dtype=dtype,
            )

    # Try to load from a string (HF repo, local path, or version name)
    if isinstance(source, (str, Path)):
        source_str = str(source)

        # Try to parse as version
        version = _try_parse_version(source_str)
        if version:
            return load_transformer(version, dtype=dtype)

        # Try to load directly from HF repo
        if _is_huggingface_repo(source_str):
            # Check if it's the 13B model repo
            if "ltxv-13b" in source_str.lower():
                # Get the right URL for the 13B model
                url = LtxvModelVersion.LTXV_13B_097_DEV.safetensors_url
                return _load_ltxv_13b_transformer(url, dtype=dtype)

            return LTXVideoTransformer3DModel.from_pretrained(
                source_str,
                subfolder="transformer",
                torch_dtype=dtype,
            )

        # Try to load from local path
        if Path(source_str).exists():
            local_path = Path(source_str)

            # Check if it's a directory with a "transformer" subfolder
            if (local_path / "transformer").exists():
                try:
                    print("从本地transformer目录加载模型...")
                    return LTXVideoTransformer3DModel.from_pretrained(
                        source_str,
                        subfolder="transformer",
                        torch_dtype=dtype,
                        local_files_only=True,
                    )
                except Exception as e:
                    print(f"从本地目录加载失败，尝试其他方法: {e}")

            # Check if it's a single .safetensors file
            if source_str.endswith(".safetensors") and local_path.is_file():
                # Check if it's the 13B model
                if "ltxv-13b" in source_str.lower():
                    return _load_ltxv_13b_transformer(source_str, dtype=dtype)

                try:
                    return LTXVideoTransformer3DModel.from_single_file(
                        source_str,
                        torch_dtype=dtype,
                    )
                except Exception as e:
                    print(f"从safetensors文件加载失败，尝试其他方法: {e}")
                    # 尝试作为UNet模型加载
                    from diffusers import UNet3DConditionModel
                    return UNet3DConditionModel.from_single_file(
                        source_str, 
                        torch_dtype=dtype
                    )
            
            # Check if it has a UNet subfolder (older diffusers format)
            elif (local_path / "unet").exists():
                print("找到unet文件夹，尝试加载UNet模型作为transformer替代...")
                from diffusers import UNet3DConditionModel
                return UNet3DConditionModel.from_pretrained(
                    str(local_path),
                    subfolder="unet",
                    torch_dtype=dtype,
                    local_files_only=True
                )
    
    raise ValueError(f"Invalid model source: {source}")

def load_ltxv_components(
    model_source: ModelSource | None = None,
    *,
    load_text_encoder_in_8bit: bool = False,
    transformer_dtype: torch.dtype = torch.float32,
    vae_dtype: torch.dtype = torch.bfloat16,
) -> LtxvModelComponents:
    """
    Load all components of the LTXV model from a specified source.
    Note: scheduler, tokenizer, and text encoder are always loaded from the main HF repo.

    Args:
        model_source: Source to load the VAE and transformer from. Can be:
            - HuggingFace repo ID (e.g. "Lightricks/LTX-Video")
            - Local path to model files (str or Path)
            - LtxvModelVersion enum value
            - None (will use the latest version)
        load_text_encoder_in_8bit: Whether to load text encoder in 8-bit precision
        transformer_dtype: Data type for transformer model
        vae_dtype: Data type for VAE model

    Returns:
        LtxvModelComponents containing all loaded model components
    """
    
    # 直接使用默认的diffusers模型路径
    if DEFAULT_DIFFUSERS_PATH.exists():
        model_source = DEFAULT_DIFFUSERS_PATH
        print(f"只使用本地diffusers模型: {DEFAULT_DIFFUSERS_PATH}")
    else:
        print(f"警告: 未找到diffusers模型路径 {DEFAULT_DIFFUSERS_PATH}")
        # 如果没有diffusers模型，再尝试使用默认版本
        if model_source is None:
            model_source = LtxvModelVersion.latest()
    
    return LtxvModelComponents(
        scheduler=load_scheduler(),
        tokenizer=load_tokenizer(),
        text_encoder=load_text_encoder(load_in_8bit=load_text_encoder_in_8bit),
        vae=load_vae(model_source, dtype=vae_dtype),
        transformer=load_transformer(model_source, dtype=transformer_dtype),
    )


def _try_parse_version(source: str | Path) -> LtxvModelVersion | None:
    """
    Try to parse a string as an LtxvModelVersion.

    Args:
        source: String to parse

    Returns:
        LtxvModelVersion if successful, None otherwise
    """
    try:
        return LtxvModelVersion(str(source))
    except ValueError:
        return None


def _is_huggingface_repo(source: str | Path) -> bool:
    """
    Check if a string is a valid HuggingFace repo ID.

    Args:
        source: String or Path to check

    Returns:
        True if the string looks like a HF repo ID
    """
    # Basic check: contains slash, no URL components
    return "/" in source and not urlparse(source).scheme


def _is_safetensors_url(source: str | Path) -> bool:
    """
    Check if a string is a valid safetensors URL.
    """
    return source.endswith(".safetensors")


def _load_ltxv_13b_transformer(safetensors_url: str, *, dtype: torch.dtype) -> LTXVideoTransformer3DModel:
    """A specific loader for LTXV-13B's transformer which doesn't yet have a Diffusers config"""
    transformer_13b_config = {
        "_class_name": "LTXVideoTransformer3DModel",
        "_diffusers_version": "0.33.0.dev0",
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 128,
        "attention_out_bias": True,
        "caption_channels": 4096,
        "cross_attention_dim": 4096,
        "in_channels": 128,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "num_attention_heads": 32,
        "num_layers": 48,
        "out_channels": 128,
        "patch_size": 1,
        "patch_size_t": 1,
        "qk_norm": "rms_norm_across_heads",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        json.dump(transformer_13b_config, f)
        f.flush()
        return LTXVideoTransformer3DModel.from_single_file(
            safetensors_url,
            config=f.name,
            torch_dtype=dtype,
        )
