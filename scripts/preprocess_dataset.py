#!/usr/bin/env python3

"""
Preprocess a video dataset by computing video clips latents and text captions embeddings.

This script provides a command-line interface for preprocessing video datasets by computing
latent representations of video clips and text embeddings of their captions. The preprocessed
data can be used to accelerate training of video generation models and to save GPU memory.

Basic usage:
    preprocess_dataset.py /path/to/dataset --resolution-buckets 768x768x49

The dataset can be either:
1. A directory containing text files with captions and video paths
2. A CSV, JSON, or JSONL file with columns for captions and video paths
"""

from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
import torchvision
import typer
from pydantic import BaseModel
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader
from transformers.utils.logging import disable_progress_bar

from ltxv_trainer.datasets import (
    PRECOMPUTED_CONDITIONS_DIR_NAME,
    PRECOMPUTED_LATENTS_DIR_NAME,
    ImageOrVideoDatasetWithResizeAndRectangleCrop,
)
from ltxv_trainer.ltxv_utils import decode_video, encode_prompt, encode_video
from ltxv_trainer.model_loader import LtxvModelVersion, load_text_encoder, load_tokenizer, load_vae

disable_progress_bar()
console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Preprocess a video dataset by computing video clips latents and text captions embeddings. "
    "The dataset can be either a directory with text files or a CSV/JSON/JSONL file.",
)

VAE_SPATIAL_FACTOR = 32
VAE_TEMPORAL_FACTOR = 8


class PreprocessingArgs(BaseModel):
    """Arguments for dataset preprocessing"""

    dataset_path: str
    caption_column: str
    video_column: str
    resolution_buckets: list[tuple[int, int, int]]
    batch_size: int
    num_workers: int
    output_dir: str | None
    id_token: str | None
    vae_tiling: bool
    decode_videos: bool


class DatasetPreprocessor:
    def __init__(self, model_source: str, device: str = "cuda", load_text_encoder_in_8bit: bool = False):
        """Initialize the preprocessor with model configuration.

        Args:
            model_source: Model source - can be a version string (e.g. "LTXV_2B_0.9.5"), HF repo, or local path
            device: Device to use for computation
            load_text_encoder_in_8bit: Whether to load text encoder in 8-bit precision
        """
        # 如果请求的是 CUDA 设备但 CUDA 不可用，默认使用 CPU
        if device.startswith('cuda') and not torch.cuda.is_available():
            print(f"\n\033[33m警告: PyTorch 没有编译 CUDA 支持或找不到可用的 CUDA 设备\033[0m")
            print(f"\033[33m自动切换到 CPU 模式\033[0m\n")
            device = 'cpu'
        
        self.device = torch.device(device)
        self._load_models(model_source, load_text_encoder_in_8bit)

    @torch.inference_mode()
    def preprocess(self, args: PreprocessingArgs) -> None:  # noqa: PLR0912
        """Run the preprocessing pipeline with the given arguments"""
        console.print("[bold blue]开始预处理...[/]")
        console.print(f"[bold cyan]使用设备: {self.device}[/]")
        console.print(f"[bold cyan]处理数据集: {args.dataset_path}[/]")
        console.print(f"[bold cyan]分辨率: {args.resolution_buckets}[/]")
        console.print(f"[bold cyan]批处理大小: {args.batch_size}[/]")
        console.print(f"[bold cyan]工作线程数: {args.num_workers}[/]")
        

        # 检查批处理大小和资源使用情况
        if args.batch_size > 1 and self.device.type == 'cpu':
            console.print("[yellow]警告: 在CPU设备上使用大于1的批处理大小可能会导致内存不足，建议使用--batch-size 1[/]")
        
        if args.batch_size > 2 and self.device.type == 'cuda':
            console.print("[yellow]警告: 在预处理高分辨率视频时，大批处理大小可能导致GPU内存不足[/]")
            console.print("[yellow]如果遇到内存错误，请尝试降低批处理大小为1或使用较低分辨率[/]")
            
        # 打印内存使用情况
        if self.device.type == 'cuda' and torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            console.print(f"[cyan]初始CUDA内存使用: {mem_allocated:.2f}MB (分配) / {mem_reserved:.2f}MB (保留)[/]")

        # Determine if dataset_path is a file or directory
        dataset_path = Path(args.dataset_path)
        is_file = dataset_path.is_file()

        # Set data_root and dataset_file based on dataset_path
        if is_file:
            data_root = str(dataset_path.parent)
            dataset_file = str(dataset_path)

            # Validate that the file exists
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

            # Validate file type
            if dataset_path.suffix.lower() not in [".csv", ".json", ".jsonl"]:
                raise ValueError(f"Dataset file must be CSV, JSON, or JSONL format: {dataset_path}")
        else:
            data_root = str(dataset_path)
            dataset_file = None

            # Validate that the directory exists
            if not dataset_path.exists() or not dataset_path.is_dir():
                raise FileNotFoundError(f"Dataset path does not exist: {dataset_path} or is not a directory")

            # Check for required files if dataset_path is a directory
            caption_file = Path(data_root) / args.caption_column
            video_file = Path(data_root) / args.video_column

            # Add .txt extension if needed
            if not caption_file.suffix:
                caption_file = caption_file.with_suffix(".txt")
                args.caption_column += ".txt"

            if not video_file.suffix:
                video_file = video_file.with_suffix(".txt")
                args.video_column += ".txt"

            # Check if caption file exists
            if not caption_file.exists():
                raise FileNotFoundError(f"Captions file does not exist: {caption_file}")

            # Check if video file exists
            if not video_file.exists():
                raise FileNotFoundError(f"Video paths file does not exist: {video_file}")

        # Create output directories
        if args.output_dir:
            output_base = Path(args.output_dir)
        else:
            # Use .precomputed directory in the dataset directory
            output_base = Path(data_root) / ".precomputed"
        
        console.print(f"[bold cyan]输出目录: {output_base}[/]")

        latents_dir, conditions_dir = self._create_output_dirs(output_base)

        if args.id_token:
            console.print(
                f"[bold yellow]LoRA trigger word[/] [cyan]{args.id_token}[/] "
                f"[bold yellow]will be prepended to all captions[/]",
            )

        # Create dataloader
        console.print("[bold cyan]创建数据加载器...[/]")
        dataloader = self._create_dataloader(
            data_root=data_root,
            dataset_file=dataset_file,
            caption_column=args.caption_column,
            video_column=args.video_column,
            resolution_buckets=args.resolution_buckets,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            id_token=args.id_token,
        )
        console.print(f"[bold cyan]数据集大小: {len(dataloader.dataset)}个样本[/]")

        # Enable/disable VAE tiling
        if args.vae_tiling:
            self.vae.enable_tiling()
        else:
            self.vae.disable_tiling()

        # Print dataset information
        console.print(f"Number of batches: {len(dataloader)} (batch size: {args.batch_size})")

        # Create progress bars
        # Process the dataset
        console.print("[bold cyan]开始处理数据集...[/]")
        console.print("[bold cyan]这可能需要较长时间，特别是对于大型视频数据集[/]")
        
        main_progress = Progress(
            TextColumn("[bold blue]预处理进度"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        with main_progress:
            task = main_progress.add_task("处理批次", total=len(dataloader))

            for batch_idx, batch in enumerate(dataloader):
                console.print(f"[cyan]处理批次 {batch_idx+1}/{len(dataloader)}[/]")
                self._process_batch(
                    batch=batch,
                    batch_idx=batch_idx,
                    batch_size=args.batch_size,
                    latents_dir=latents_dir,
                    conditions_dir=conditions_dir,
                    output_base=output_base,
                    decode_videos=args.decode_videos,
                )
                main_progress.update(task, advance=1)
                console.print(f"[green]批次 {batch_idx+1} 处理完成[/]")

        # Print summary
        console.print(
            f"[bold green]✓[/] Processed [bold]{len(dataloader.dataset)}[/] items. "
            f"Results saved to [cyan]{output_base}[/]",
        )

    def _load_models(self, model_source: str, load_text_encoder_in_8bit: bool) -> None:
        """Initialize and load the required models"""
        with console.status(f"[bold]Loading models from [cyan]{model_source}[/]...", spinner="dots"):
            # 在离线模式下使用虚拟VAE
            try:
                # 加载VAE模型并移动到指定设备
                self.vae = load_vae(model_source, dtype=torch.bfloat16)
                # 安全地移动到设备
                try:
                    self.vae = self.vae.to(self.device)
                except Exception as e:
                    print(f"\n\033[33m警告: 无法将VAE移动到{self.device}设备: {e}\033[0m")
                    print(f"\033[33m保持VAE在CPU上运行\033[0m\n")
                
                # 加载分词器
                self.tokenizer = load_tokenizer()
                
                # 加载文本编码器
                self.text_encoder = load_text_encoder(load_in_8bit=load_text_encoder_in_8bit)
                # 安全地移动到设备
                try:
                    self.text_encoder = self.text_encoder.to(self.device)
                except Exception as e:
                    print(f"\n\033[33m警告: 无法将文本编码器移动到{self.device}设备: {e}\033[0m")
                    print(f"\033[33m保持文本编码器在CPU上运行\033[0m\n")
                    
            except Exception as e:
                print(f"\n\033[31m错误: 模型加载失败: {e}\033[0m")
                raise

        console.print("[bold green]✓[/] Models loaded successfully")

    @staticmethod
    def _create_output_dirs(output_base: Path) -> tuple[Path, Path]:
        """Create and return paths for output directories"""
        latents_dir = output_base / PRECOMPUTED_LATENTS_DIR_NAME
        conditions_dir = output_base / PRECOMPUTED_CONDITIONS_DIR_NAME

        latents_dir.mkdir(parents=True, exist_ok=True)
        conditions_dir.mkdir(parents=True, exist_ok=True)

        return latents_dir, conditions_dir

    @staticmethod
    def _create_dataloader(
        data_root: str,
        dataset_file: str | None,
        caption_column: str,
        video_column: str,
        resolution_buckets: list[tuple[int, int, int]],
        batch_size: int,
        num_workers: int,
        id_token: str | None,
    ) -> DataLoader:
        """Initialize dataset and create dataloader"""
        with console.status("[bold]Loading dataset...", spinner="dots"):
            dataset = ImageOrVideoDatasetWithResizeAndRectangleCrop(
                data_root=data_root,
                dataset_file=dataset_file,
                caption_column=caption_column,
                video_column=video_column,
                resolution_buckets=resolution_buckets,
                id_token=id_token,
            )

        console.print(f"[bold green]✓[/] Dataset loaded with [bold]{len(dataset)}[/] items")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

    def _process_batch(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        batch_size: int,
        latents_dir: Path,
        conditions_dir: Path,
        output_base: Path,
        decode_videos: bool,
    ) -> None:
        """Process a single batch of data and save the results"""
        console.print(f"[cyan]详细处理批次 {batch_idx+1}...[/]")
        
        # 先创建调试日志目录
        debug_dir = output_base / "debug_logs"
        debug_dir.mkdir(exist_ok=True, parents=True)
        error_log = debug_dir / f"error_batch_{batch_idx}.txt"
        
        try:
            # 增加更多调试信息
            console.print(f"[bold blue]===== 批次处理调试信息 =====\n")
            console.print(f"[yellow]批次索引: {batch_idx}")
            console.print(f"[yellow]批次大小: {batch_size}")
            console.print(f"[yellow]批次键值: {list(batch.keys())}[/]")
            
            # 保存批次调试信息
            debug_file = debug_dir / f"batch_{batch_idx}_debug.txt"
            try:
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(f"Batch keys: {list(batch.keys())}\n")
                    for k, v in batch.items():
                        try:
                            f.write(f"Key: {k}, Type: {type(v)}\n")
                            if isinstance(v, dict):
                                f.write(f"\tDict keys: {list(v.keys())}\n")
                            elif isinstance(v, torch.Tensor):
                                f.write(f"\tTensor shape: {v.shape}, dtype: {v.dtype}, device: {v.device}\n")
                        except Exception as debug_err:
                            f.write(f"Could not debug key: {k}, error: {str(debug_err)}\n")
                console.print(f"[green]调试信息已保存到: {debug_file}[/]")
            except Exception as e:
                console.print(f"[red]无法保存调试信息: {str(e)}[/]")
            
            # 如果有视频元数据，输出它
            if "video_metadata" in batch:
                try:
                    console.print(f"[yellow]视频元数据: {batch['video_metadata']}[/]")
                except Exception as e:
                    console.print(f"[red]无法显示视频元数据: {str(e)}[/]")
            
            console.print(f"[bold blue]===== 调试信息结束 =====\n")
        except Exception as batch_error:
            error_message = f"\u5904理批次 {batch_idx} 时出错: {str(batch_error)}"
            console.print(f"[bold red]{error_message}[/]")
            # 将错误信息写入日志文件
            try:
                with open(error_log, "w", encoding="utf-8") as f:
                    f.write(f"Error processing batch {batch_idx}:\n")
                    f.write(f"Error message: {str(batch_error)}\n")
                    import traceback
                    f.write(f"Traceback:\n{traceback.format_exc()}\n")
                    f.write(f"Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'Not a dict'}\n")
                console.print(f"[yellow]错误日志已保存到: {error_log}[/]")
            except Exception as log_error:
                console.print(f"[bold red]无法保存错误日志: {str(log_error)}[/]")
            # 把错误往上抛出，让程序可以正常终止
            raise
        
        # 兼容不同的键名
        if "pixel_values" in batch:
            pixel_values = batch["pixel_values"]  # Shape: [B, F, C, H, W]
        elif "video" in batch:
            pixel_values = batch["video"]
        else:
            # 尝试找到可能的视频数据键
            video_keys = [k for k in batch.keys() if any(v in k.lower() for v in ["video", "image", "pixel", "frame"])]
            if video_keys:
                pixel_values = batch[video_keys[0]]
                console.print(f"[yellow]使用替代键 '{video_keys[0]}' 替代 'pixel_values'[/]")
            else:
                raise KeyError(f"Cannot find video data in batch with keys: {list(batch.keys())}")
        
        # 获取提示文本
        if "prompt" in batch:
            prompts = batch["prompt"]
        elif "caption" in batch:
            prompts = batch["caption"]
        else:
            # 尝试找到可能的文本数据键
            text_keys = [k for k in batch.keys() if any(v in k.lower() for v in ["prompt", "caption", "text", "description"])]
            if text_keys:
                prompts = batch[text_keys[0]]
                console.print(f"[yellow]使用替代键 '{text_keys[0]}' 替代 'prompt'[/]")
            else:
                raise KeyError(f"Cannot find text data in batch with keys: {list(batch.keys())}")
                
        # 获取元数据
        if "video_metadata" in batch:
            video_metadata = batch["video_metadata"]
        else:
            # 如果没有元数据，创建一个默认的
            console.print(f"[yellow]未找到视频元数据，使用默认值[/]")
            video_metadata = {"fps": [30.0] * len(prompts)}
            
        console.print(f"[cyan]批次大小: {len(prompts)}个样本[/]")

        # Generate indices for each item in the batch
        if batch_idx == 0:
            first_idx = 0
        else:
            first_idx = batch_idx * batch_size

        # Encode prompts
        text_embeddings = encode_prompt(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            prompt=batch["prompt"],
            device=self.device,
        )

        # Encode videos/images
        try:
            # 尝试清理一下GPU内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            # 打印内存使用情况
            if self.device.type == 'cuda' and torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated(self.device) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(self.device) / 1024**2
                console.print(f"[yellow]当前CUDA内存使用: {mem_allocated:.2f}MB (分配) / {mem_reserved:.2f}MB (保留)[/]")
                
            # 提示当前使用的设备
            console.print(f"[bold green]使用设备: {self.device}[/]")
            
            console.print(f"[cyan]开始编码视频，形状: {batch['video'].shape}[/]")
            video_latents = encode_video(
                vae=self.vae,
                image_or_video=batch["video"],
                device=self.device,
            )
            console.print(f"[green]视频编码完成，潜在空间形状: {video_latents['latents'].shape}[/]")
            
        except torch.cuda.OutOfMemoryError as oom_error:
            error_msg = f"CUDA内存不足! 请尝试减小批处理大小或视频分辨率。错误: {str(oom_error)}"
            console.print(f"[bold red]{error_msg}[/]")
            with open(error_log, "w", encoding="utf-8") as f:
                f.write(f"{error_msg}\n")
                f.write("解决方法:\n")
                f.write("1. 减小批处理大小 (--batch-size 1)\n")
                f.write("2. 减小视频分辨率 (例如 256x256 而不是 512x512)\n")
                f.write("3. 使用CPU处理 (--device cpu)\n")
            raise
            
        except Exception as encode_error:
            error_msg = f"视频编码失败: {str(encode_error)}"
            console.print(f"[bold red]{error_msg}[/]")
            with open(error_log, "w", encoding="utf-8") as f:
                f.write(f"{error_msg}\n")
                import traceback
                f.write(f"堆栈跟踪:\n{traceback.format_exc()}")
                if isinstance(batch["video"], torch.Tensor):
                    f.write(f"\n视频形状: {batch['video'].shape}\n")
                    f.write(f"视频类型: {batch['video'].dtype}\n")
                    f.write(f"视频设备: {batch['video'].device}\n")
            raise

        # Save each item in the batch
        for i in range(len(batch["prompt"])):
            file_idx = batch_idx * batch_size + i
            latent_path = latents_dir / f"latent_{file_idx:08d}.pt"
            condition_path = conditions_dir / f"condition_{file_idx:08d}.pt"

            fps = batch["video_metadata"]["fps"][i].item()
            # 尝试清理一下内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            latent_item = {
                "latents": video_latents["latents"][i].cpu().contiguous(),
                "num_frames": video_latents["num_frames"],
                "height": video_latents["height"],
                "width": video_latents["width"],
                "fps": fps,
            }
            condition_item = {
                "prompt_embeds": text_embeddings["prompt_embeds"][i].cpu().contiguous(),
                "prompt_attention_mask": text_embeddings["prompt_attention_mask"][i].cpu().contiguous(),
            }

            torch.save(latent_item, latent_path)
            torch.save(condition_item, condition_path)

            # Decode video/image if requested
            if decode_videos:
                decoded_dir = output_base / "decoded_videos"
                decoded_dir.mkdir(parents=True, exist_ok=True)

                video = decode_video(
                    vae=self.vae,
                    latents=latent_item["latents"],
                    num_frames=latent_item["num_frames"],
                    height=latent_item["height"],
                    width=latent_item["width"],
                    device=self.device,
                )
                video = video[0]  # Remove batch dimension
                # Convert to uint8 for saving
                video = (video * 255).round().clamp(0, 255).to(torch.uint8)
                video = video.permute(1, 2, 3, 0)  # [C,F,H,W] -> [F,H,W,C]

                # For single frame (images), save as PNG, otherwise as MP4
                is_image = video.shape[0] == 1
                if is_image:
                    output_path = decoded_dir / f"image_{file_idx:08d}.png"
                    torchvision.utils.save_image(
                        video[0].permute(2, 0, 1) / 255.0,  # [H,W,C] -> [C,H,W] and normalize
                        str(output_path),
                    )
                else:
                    output_path = decoded_dir / f"video_{file_idx:08d}.mp4"
                    torchvision.io.write_video(
                        str(output_path),
                        video.cpu(),
                        fps=Fraction(fps).limit_denominator(1000),
                        video_codec="h264",
                        options={"crf": "18"},
                    )


def _parse_resolution_buckets(resolution_buckets_str: str, frames: int = None) -> list[tuple[int, int, int]]:
    """Parse resolution buckets from string format to list of tuples
    
    Args:
        resolution_buckets_str: Resolution buckets in string format
        frames: Optional frame count to use if not specified in resolution
        
    Returns:
        List of tuples in (frames, height, width) format
    """
    resolution_buckets = []
    for bucket_str in resolution_buckets_str.split(";"):
        # 检查分辨率格式
        parts = bucket_str.split("x")
        
        if len(parts) == 3:  # 标准WxHxF格式
            w, h, f = map(int, parts)
            print(f"使用完整的分辨率格式: {bucket_str}")
        elif len(parts) == 2 and frames:  # WxH格式 + 单独的frames参数
            w, h = map(int, parts)
            f = frames
            print(f"将WxH格式与帧数参数结合: {w}x{h}x{f}")
        elif len(parts) == 2:  # WxH格式但无frames参数
            w, h = map(int, parts)
            f = 49  # 默认帧数
            print(f"使用默认帧数(49): {w}x{h}x{f}")
        else:
            raise typer.BadParameter(
                f"Invalid resolution format: {bucket_str}. Must be WxH or WxHxF",
                param_hint="resolution-buckets",
            )

        if w % VAE_SPATIAL_FACTOR != 0 or h % VAE_SPATIAL_FACTOR != 0:
            raise typer.BadParameter(
                f"Width and height must be multiples of {VAE_SPATIAL_FACTOR}, got {w}x{h}",
                param_hint="resolution-buckets",
            )

        if f % VAE_TEMPORAL_FACTOR != 1:
            raise typer.BadParameter(
                f"Number of frames must be a multiple of {VAE_TEMPORAL_FACTOR} plus 1, got {f}",
                param_hint="resolution-buckets",
            )

        resolution_buckets.append((f, h, w))
    return resolution_buckets


@app.command()
def main(  # noqa: PLR0913
    dataset_path: str = typer.Argument(
        ...,
        help="Path to dataset directory or metadata file (CSV/JSON/JSONL)",
    ),
    resolution_buckets: str = typer.Option(
        ...,
        help='Resolution buckets in format "WxH;WxH;..." or "WxHxF;WxHxF;..." (e.g. "768x768" or "768x768x25")',
    ),
    caption_column: str = typer.Option(
        default="caption",
        help="Column name or filename for captions: "
        "If dataset_path is a CSV/JSON/JSONL file, this is the column name containing captions. "
        "If dataset_path is a directory, this is the filename containing line-separated captions.",
    ),
    video_column: str = typer.Option(
        default="media_path",
        help="Column name or filename for videos: "
        "If dataset_path is a CSV/JSON/JSONL file, this is the column name containing video paths. "
        "If dataset_path is a directory, this is the filename containing line-separated video paths.",
    ),
    batch_size: int = typer.Option(
        default=1,
        help="Batch size for preprocessing",
    ),
    num_workers: int = typer.Option(
        default=1,
        help="Number of dataloader workers",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    load_text_encoder_in_8bit: bool = typer.Option(
        default=False,
        help="Load the T5 text encoder in 8-bit precision to save memory",
    ),
    vae_tiling: bool = typer.Option(
        default=False,
        help="Enable VAE tiling for larger video resolutions",
    ),
    output_dir: str | None = typer.Option(
        default=None,
        help="Output directory (defaults to .precomputed in dataset directory)",
    ),
    model_source: str = typer.Option(
        default=str(LtxvModelVersion.latest()),
        help="Model source - can be a version string (e.g. 'LTXV_2B_0.9.5'), HF repo, or local path",
    ),
    id_token: str | None = typer.Option(
        default=None,
        help="Optional token to prepend to each caption (acts as a trigger word when training a LoRA)",
    ),
    decode_videos: bool = typer.Option(
        default=False,
        help="Decode and save videos after encoding (for verification purposes)",
    ),
    frames: int | None = typer.Option(
        default=None,
        help="Number of frames to use if not specified in resolution buckets. Must be multiple of 8 plus 1 (e.g., 25, 49, 73).",
    ),
) -> None:
    """Preprocess a video dataset by computing and saving latents and text embeddings.

    The dataset can be specified in two ways:
    1. A directory containing text files with captions and video paths
    2. A CSV, JSON, or JSONL file with columns for captions and video paths
    """
    parsed_resolution_buckets = _parse_resolution_buckets(resolution_buckets, frames)

    if len(parsed_resolution_buckets) > 1:
        raise typer.BadParameter(
            "Multiple resolution buckets are not yet supported. Please specify only one bucket.",
            param_hint="resolution-buckets",
        )

    try:
        import datetime
        import os
        import traceback
        
        # 保存调试信息
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            debug_dir = os.path.join(output_dir, "debug_logs")
            os.makedirs(debug_dir, exist_ok=True)
            
            debug_file = os.path.join(debug_dir, f"params_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(f"预处理参数 ({datetime.datetime.now()}):\n")
                f.write(f"dataset_path: {dataset_path}\n")
                f.write(f"resolution_buckets: {resolution_buckets}\n")
                f.write(f"caption_column: {caption_column}\n")
                f.write(f"video_column: {video_column}\n")
                f.write(f"batch_size: {batch_size}\n")
                f.write(f"num_workers: {num_workers}\n")
                f.write(f"device: {device}\n")
                f.write(f"load_text_encoder_in_8bit: {load_text_encoder_in_8bit}\n")
                f.write(f"vae_tiling: {vae_tiling}\n")
                f.write(f"output_dir: {output_dir}\n")
                f.write(f"model_source: {model_source}\n")
                f.write(f"id_token: {id_token}\n")
                f.write(f"decode_videos: {decode_videos}\n")
            console.print(f"[green]调试信息已保存到 {debug_file}[/]")
        
        # 验证media_path.txt
        if os.path.isdir(dataset_path):
            media_path_file = os.path.join(dataset_path, video_column + ".txt")
            if not os.path.exists(media_path_file):
                console.print(f"[bold red]错误: 找不到视频路径文件 {media_path_file}[/]")
                console.print("请确保在数据集目录中创建了包含视频文件名的media_path.txt文件")
                return
            
            # 检查视频文件是否存在
            with open(media_path_file, "r", encoding="utf-8") as f:
                video_files = [line.strip() for line in f.readlines()]
            
            missing_videos = []
            for video in video_files:
                video_path = os.path.join(dataset_path, video)
                if not os.path.exists(video_path):
                    missing_videos.append(video)
            
            if missing_videos:
                console.print(f"[bold red]警告: {len(missing_videos)}/{len(video_files)} 个视频文件不存在:[/]")
                for missing in missing_videos[:5]:  # 只显示前5个
                    console.print(f"  - {missing}")
                if len(missing_videos) > 5:
                    console.print(f"  ...以及其他 {len(missing_videos)-5} 个文件")
                console.print("请确保所有视频文件都在数据集根目录中")
        
        # 构建预处理参数
        args = PreprocessingArgs(
            dataset_path=dataset_path,
            caption_column=caption_column,
            video_column=video_column,
            resolution_buckets=parsed_resolution_buckets,
            batch_size=batch_size,
            num_workers=num_workers,
            output_dir=output_dir,
            id_token=id_token,
            vae_tiling=vae_tiling,
            decode_videos=decode_videos,
        )

        # 初始化预处理器并执行预处理
        preprocessor = DatasetPreprocessor(
            model_source=model_source,
            device=device,
            load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        )
        preprocessor.preprocess(args)
        
    except Exception as e:
        console.print(f"[bold red]预处理过程中发生错误:[/]")
        console.print(f"[red]{str(e)}[/]")
        # 保存详细的错误信息
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            error_file = os.path.join(output_dir, "error_log.txt")
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(f"预处理错误 ({datetime.datetime.now()}):\n")
                f.write(f"错误信息: {str(e)}\n")
                f.write("详细堆栈:\n")
                f.write(traceback.format_exc())
            console.print(f"[yellow]详细错误日志已保存到 {error_file}[/]")
        raise

if __name__ == "__main__":
    app()
