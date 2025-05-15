#!/usr/bin/env python

"""
Train LTXV models using configuration from YAML files.

This script provides a command-line interface for training LTXV models using
either LoRA fine-tuning or full model fine-tuning. It loads configuration from
a YAML file and passes it to the trainer.

Basic usage:
    train.py --config configs/ltxv_lora_config.yaml
"""

import os
from pathlib import Path

import typer
import yaml
from rich.console import Console

from ltxv_trainer.config import LtxvTrainerConfig
from ltxv_trainer.trainer import LtxvTrainer

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

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Train LTXV models using configuration from YAML files.",
)


@app.command()
def main(config_path: str = typer.Argument(..., help="Path to YAML configuration file")) -> None:
    """Train the model using the provided configuration file."""
    # Load the configuration from the YAML file
    config_path = Path(config_path)
    if not config_path.exists():
        typer.echo(f"Error: Configuration file {config_path} does not exist.")
        raise typer.Exit(code=1)

    with open(config_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    # Convert the loaded data to the LtxvTrainerConfig object
    try:
        trainer_config = LtxvTrainerConfig(**config_data)
    except Exception as e:
        typer.echo(f"Error: Invalid configuration data: {e}")
        raise typer.Exit(code=1) from e

    # Initialize the training process
    trainer = LtxvTrainer(trainer_config)
    trainer.train()


if __name__ == "__main__":
    app()
