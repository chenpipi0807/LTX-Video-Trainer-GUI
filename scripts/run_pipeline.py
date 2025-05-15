#!/usr/bin/env python3

"""
Master script to run the complete LTXV LoRA training pipeline.

This script orchestrates the entire pipeline:
1. Scene splitting (if raw videos exist)
2. Video captioning (if scenes exist)
3. Dataset preprocessing
4. Model training

Usage:
    run_pipeline.py basename --resolution-buckets 768x768x49 --config-template configs/ltxv_lora_config.yaml
"""

import inspect
import os

# Add the parent directory to the Python path so we can import from scripts
import sys
from pathlib import Path
from typing import Callable

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typer.models import OptionInfo

# Add the parent directory to the Python path so we can import from scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

# 检查是否在离线模式运行
OFFLINE_MODE = os.environ.get("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes", "on")
if OFFLINE_MODE:
    console = Console()
    console.print("[bold blue]检测到离线模式[/]")
    console.print("[bold green]使用diffusers直接加载本地模型，不需要适配器[/]")
    
    # 设置环境变量以强制离线模式
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

from scripts.caption_videos import VIDEO_EXTENSIONS
from scripts.caption_videos import main as caption_videos
from scripts.convert_checkpoint import main as convert_checkpoint
from scripts.preprocess_dataset import main as preprocess_dataset
from scripts.split_scenes import main as split_scenes
from scripts.train import main as train


def typer_unpacker(f: Callable) -> Callable:
    def wrapper(*args, **kwargs) -> None:
        # Get the default function argument that aren't passed in kwargs via the
        # inspect module: https://stackoverflow.com/a/12627202
        missing_default_values = {
            k: v.default
            for k, v in inspect.signature(f).parameters.items()
            if v.default is not inspect.Parameter.empty and k not in kwargs
        }

        for name, func_default in missing_default_values.items():
            # If the default value is a typer.Option or typer.Argument, we have to
            # pull either the .default attribute and pass it in the function
            # invocation, or call it first.
            if isinstance(func_default, OptionInfo):
                if callable(func_default.default):
                    kwargs[name] = func_default.default()
                else:
                    kwargs[name] = func_default.default

        # Call the wrapped function with the defaults injected if not specified.
        return f(*args, **kwargs)

    return wrapper


console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Run the complete LTXV training pipeline.",
)


def process_raw_videos(raw_dir: Path, scenes_dir: Path) -> None:
    """Process raw videos by splitting them into scenes.

    Args:
        raw_dir: Directory containing raw videos
        scenes_dir: Directory to save split scenes
    """
    # Get all video files
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(list(raw_dir.glob(f"*.{ext}")) + list(raw_dir.glob(f"*.{ext.upper()}")))

    if not video_files:
        console.print("[bold yellow]No video files found in raw directory.[/]")
        return

    console.print(f"Found [bold]{len(video_files)}[/] video files to process.")

    # Create scenes directory
    scenes_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if scenes have already been generated (directories match expected pattern)
    scene_dirs = list(scenes_dir.glob("scene_*"))
    if scene_dirs:
        console.print(f"[bold yellow]Found {len(scene_dirs)} existing scene directories in {scenes_dir}.[/]")
        console.print("[bold yellow]Skipping scene detection to avoid redundant processing.[/]")
        console.print("[bold yellow]If you want to reprocess videos, delete the scene directories first.[/]")
        return

    # Get the main function from the registered commands
    split_func = typer_unpacker(split_scenes)

    # Process each video
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Splitting videos into scenes", total=len(video_files))

        for video_file in video_files:
            # Split the video into scenes directly in the scenes directory
            console.print(f"Splitting video: {video_file}")
            split_func(
                video_path=str(video_file),
                output_dir=str(scenes_dir),
                detector="content",  # Use default content-based detection
            )

            progress.advance(task)


def process_scenes(scenes_dir: Path) -> None:
    """Process scenes by generating captions.

    Args:
        scenes_dir: Directory containing split scenes
    """
    # Check if scenes directory exists and contains subdirectories
    if not scenes_dir.exists() or not any(scenes_dir.iterdir()):
        console.print("[bold yellow]No scenes directory found or empty.[/]")
        return
        
    # 检查是否已有标题文件
    captions_file = scenes_dir / "captions.json"
    if captions_file.exists():
        console.print(f"[bold yellow]找到现有标题文件: {captions_file}[/]")
        console.print("[bold yellow]跳过生成标题步骤以避免重复处理。[/]")
        console.print("[bold yellow]如果要重新生成标题，请删除该文件。[/]")
        return

    # Get the main function from the registered commands
    caption_func = typer_unpacker(caption_videos)
    
    # 使用改进的错误处理机制尝试生成标题
    try:
        caption_func(
            input_path=str(scenes_dir),  # Use current directory (scenes_dir)
            output=str(scenes_dir / "captions.json"),  # Save in current directory
            captioner_type="llava_next_7b",  # Use default captioner
        )
    except Exception as e:
        console.print(f"[bold red]生成视频标题时出错: {e}[/]")
        console.print("[bold yellow]请确保已安装所有必要的库依赖[/]")
        raise


def preprocess_data(scenes_dir: Path, resolution_buckets: str, id_token: str | None = None) -> None:
    """Preprocess the dataset using the provided resolution buckets.

    Args:
        scenes_dir: Directory containing split scenes and captions
        resolution_buckets: Resolution buckets string (e.g. "768x768x49")
        id_token: Optional token to prepend to each caption (acts as a trigger word when training a LoRA)
    """
    if not scenes_dir.exists():
        console.print("[bold yellow]Scenes directory not found.[/]")
        return

    # Check for captions file
    captions_file = scenes_dir / "captions.json"
    if not captions_file.exists():
        console.print("[bold yellow]Captions file not found.[/]")
        return

    # Create preprocessed data directory
    preprocessed_dir = scenes_dir / ".precomputed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    # Get the main function from the registered commands
    preprocess_func = typer_unpacker(preprocess_dataset)

    # Preprocess the dataset
    preprocess_func(
        dataset_path=str(captions_file),
        resolution_buckets=resolution_buckets,
        caption_column="caption",
        video_column="media_path",
        output_dir=str(preprocessed_dir),
        id_token=id_token,
        num_workers=0,  # 禁用多进程以避免序列化错误
        decode_videos=False,  # 禁用视频解码以避免torchvision错误
    )


def prepare_and_run_training(
    basename: str,
    config_template: Path,
    scenes_dir: Path,
    rank: int,
    trigger_word: str | None = None,
) -> None:
    """Prepare training configuration and run training.

    Args:
        basename: Base name for the project
        config_template: Path to the configuration template file
        scenes_dir: Directory containing preprocessed data
        rank: LoRA rank to use for training
    """
    if not config_template.exists():
        console.print(f"[bold red]Configuration template not found: {config_template}[/]")
        return

    # Read template and replace placeholders
    # 显式指定UTF-8编码，避免使用系统默认编码（如GBK）
    config_content = config_template.read_text(encoding='utf-8')
    config_content = config_content.replace("[BASENAME]", basename)
    config_content = config_content.replace("[RANK]", str(rank))

    # Parse the config content to get the output directory
    import json
    import yaml

    try:
        config_data = yaml.safe_load(config_content)
        if config_data is None:
            # 处理空YAML文件的情况
            console.print("[bold red]配置模板内容为空或无效，使用默认配置[/]")
            config_data = {}
    except Exception as e:
        console.print(f"[bold red]无法解析YAML配置: {str(e)}，使用默认配置[/]")
        config_data = {}
        
    output_dir = Path(config_data.get("output_dir", f"outputs/{basename}_training"))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read prompts from captions.json if available
    captions_file = scenes_dir / "captions.json"
    if captions_file.exists():
        with open(captions_file) as f:
            captions_data = json.load(f)
            # Get up to 3 prompts from the captions
            prompts = [item["caption"] for item in captions_data[:3]]
            if prompts and trigger_word:
                # 使用提取的触发词创建提示词
                custom_prompts = []
                for prompt in prompts[:3]:  # 限制只使用3个提示词
                    # 先移除原有的触发词前缀（如果有）
                    words = prompt.split()
                    if len(words) > 1:
                        # 假设第一个词是触发词，移除它
                        content = " ".join(words[1:])
                    else:
                        content = prompt
                    
                    # 添加新的触发词
                    custom_prompts.append(f"{trigger_word} {content}")
                
                # 确保validation键存在
                if "validation" not in config_data:
                    config_data["validation"] = {}
                # Replace validation.prompts in the config
                config_data["validation"]["prompts"] = custom_prompts
                console.print(f"[bold blue]使用自定义触发词 '{trigger_word}' 生成验证提示: {custom_prompts}[/]")
            elif prompts:
                # 如果没有触发词，使用原始提示词
                # 确保validation键存在
                if "validation" not in config_data:
                    config_data["validation"] = {}
                # Replace validation.prompts in the config
                config_data["validation"]["prompts"] = prompts
                # Convert back to YAML string
                config_content = yaml.dump(config_data)

    # Save instantiated configuration
    config_path = output_dir / "config.yaml"
    config_path.write_text(config_content, encoding='utf-8')

    # Get the main function from the registered commands
    train_func = typer_unpacker(train)

    # Run training
    train_func(config_path=str(config_path))

    # Convert LoRA to ComfyUI format
    console.print("[bold blue]Converting LoRA to ComfyUI format...[/]")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Find the latest checkpoint in the output directory
    lora_checkpoints = list(checkpoint_dir.glob("lora_weights_step_*.safetensors"))
    if not lora_checkpoints:
        console.print("[bold yellow]No LoRA checkpoints found in output directory.[/]")
        lora_path = None
    else:
        # Sort by step number (extracted from filename)
        lora_checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
        lora_path = lora_checkpoints[-1]  # Get the latest checkpoint
        console.print(f"[bold blue]Found latest checkpoint: {lora_path.name}[/]")

    if lora_path and lora_path.exists():
        convert_func = typer_unpacker(convert_checkpoint)
        convert_func(
            input_path=str(lora_path),
            to_comfy=True,
        )
        console.print("[bold green]LoRA conversion complete![/]")
    else:
        console.print(f"[bold yellow]No LoRA weights found at {lora_path} to convert.[/]")


@app.command()
def main(
    basename: str = typer.Argument(..., help="Base name for the project"),
    resolution_buckets: str = typer.Option(
        ...,
        help='Resolution buckets in format "WxHxF" (e.g. "768x768x49")',
    ),
    config_template: Path = typer.Option(  # noqa: B008
        ...,
        help="Path to the configuration template file",
        exists=True,
        dir_okay=False,
    ),
    id_token: str | None = typer.Option(
        default=None,
        help="Optional token to prepend to each caption (acts as a trigger word when training a LoRA)",
    ),
    trigger_word: str | None = typer.Option(
        default=None,
        help="Trigger word to use in validation prompts (automatically extracted from basename if not provided)",
    ),
    rank: int = typer.Option(
        ...,
        help="LoRA rank to use for training",
        min=1,
        max=128,
    ),
) -> None:
    """Run the complete LTXV training pipeline."""
    # Define directories
    raw_dir = Path(f"{basename}_raw")
    scenes_dir = Path(f"{basename}_scenes")

    # Step 1: Process raw videos if they exist
    if raw_dir.exists() and any(raw_dir.iterdir()):
        console.print("[bold blue]Step 1: Processing raw videos...[/]")
        process_raw_videos(raw_dir, scenes_dir)
    else:
        console.print("[bold yellow]Raw videos directory not found or empty. Skipping scene splitting.[/]")

    # Step 2: Generate captions if scenes exist
    if scenes_dir.exists() and any(scenes_dir.iterdir()):
        console.print("[bold blue]Step 2: Generating captions...[/]")
        process_scenes(scenes_dir)
    else:
        console.print("[bold yellow]Scenes directory not found or empty. Skipping captioning.[/]")

    # 如果没有提供触发词，尝试从分支名提取
    if trigger_word is None:
        # 简单的提取逻辑
        if "_" in basename:
            trigger_word = basename.split("_")[0]
        elif " " in basename:
            trigger_word = basename.split(" ")[0]
        else:
            trigger_word = basename
        console.print(f"[bold blue]从项目名称 '{basename}' 中提取触发词: '{trigger_word}'[/]")
    
    # Step 3: Preprocess dataset
    console.print("[bold blue]Step 3: Preprocessing dataset...[/]")
    preprocess_data(scenes_dir, resolution_buckets, id_token)

    # Step 4: Run training
    console.print("[bold blue]Step 4: Running training...[/]")
    prepare_and_run_training(basename, config_template, scenes_dir, rank, trigger_word)

    console.print("[bold green]Pipeline completed successfully![/]")


if __name__ == "__main__":
    app()
