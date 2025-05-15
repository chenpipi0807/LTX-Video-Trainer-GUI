#!/usr/bin/env python3

"""
Decode precomputed video latents back into videos using the VAE.

This script loads latent files saved during preprocessing and decodes them
back into video clips using the same VAE model.

Basic usage:
    decode_latents.py /path/to/latents/dir --output-dir /path/to/output
"""

from pathlib import Path

import torch
import torchvision
import typer
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
from transformers.utils.logging import disable_progress_bar

from ltxv_trainer.ltxv_utils import decode_video
from ltxv_trainer.model_loader import LtxvModelVersion, load_vae

disable_progress_bar()
console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Decode precomputed video latents back into videos using the VAE.",
)


class LatentsDecoder:
    def __init__(self, model_source: str, device: str = "cuda"):
        """Initialize the decoder with model configuration.

        Args:
            model_source: Model source - can be a version string, HF repo, or local path
            device: Device to use for computation
        """
        self.device = torch.device(device)
        self._load_model(model_source)

    def _load_model(self, model_source: str) -> None:
        """Initialize and load the VAE model"""
        with console.status("[bold]Loading VAE model...", spinner="dots"):
            console.print(f"Loading VAE from [cyan]{model_source}[/]")
            self.vae = load_vae(model_source, dtype=torch.bfloat16).to(self.device)

        console.print("[bold green]✓[/] VAE model loaded successfully")

    @torch.inference_mode()
    def decode(self, latents_dir: Path, output_dir: Path, seed: int | None = None) -> None:
        """Decode all latent files in the given directory"""
        output_dir.mkdir(parents=True, exist_ok=True)
        latent_files = sorted(latents_dir.glob("*.pt"))

        if not latent_files:
            console.print(f"[bold red]No latent files found in {latents_dir}[/]")
            return

        console.print(f"Found [bold]{len(latent_files)}[/] latent files to decode")

        # Process files one by one
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            task = progress.add_task(
                "Decoding latents",
                total=len(latent_files),
            )

            for latent_file in latent_files:
                self._process_file(latent_file, output_dir, seed)
                progress.advance(task)

        console.print(
            f"[bold green]✓[/] Decoded [bold]{len(latent_files)}[/] latent files. "
            f"Results saved to [cyan]{output_dir}[/]",
        )

    def _process_file(self, latent_file: Path, output_dir: Path, seed: int | None) -> None:
        """Process a single latent file"""
        # Load the latent data
        data = torch.load(latent_file, map_location=self.device)

        # Create generator only if seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        # Decode the video
        video = decode_video(
            vae=self.vae,
            latents=data["latents"],
            num_frames=data["num_frames"],
            height=data["height"],
            width=data["width"],
            device=self.device,
            patch_size=1,  # These match the transformer's config in the pipeline
            patch_size_t=1,
            generator=generator,
        )

        video = video[0]  # Remove batch dimension

        # Convert to uint8 for saving
        video = (video * 255).round().clamp(0, 255).to(torch.uint8)
        video = video.permute(1, 2, 3, 0)  # [C,F,H,W] -> [F,H,W,C]

        # Save as video file
        output_path = output_dir / f"{latent_file.stem}.mp4"
        torchvision.io.write_video(
            str(output_path),
            video.cpu(),
            # TODO: take the fps from a stored value in the latent file
            fps=30,
            video_codec="h264",
            options={"crf": "18"},
        )


@app.command()
def main(
    latents_dir: str = typer.Argument(
        ...,
        help="Directory containing the precomputed latent files",
    ),
    output_dir: str = typer.Option(
        ...,
        help="Directory to save the decoded videos",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    model_source: str = typer.Option(
        default=str(LtxvModelVersion.latest()),
        help="Model source - can be a version string (e.g. 'LTXV_2B_0.9.5'), HF repo, or local path",
    ),
    seed: int = typer.Option(
        default=None,
        help="Random seed for noise generation",
    ),
) -> None:
    """Decode precomputed video latents back into videos using the VAE."""
    latents_path = Path(latents_dir)
    output_path = Path(output_dir)

    if not latents_path.exists() or not latents_path.is_dir():
        raise typer.BadParameter(f"Latents directory does not exist: {latents_path}")

    decoder = LatentsDecoder(model_source=model_source, device=device)
    decoder.decode(latents_path, output_path, seed=seed)


if __name__ == "__main__":
    app()
