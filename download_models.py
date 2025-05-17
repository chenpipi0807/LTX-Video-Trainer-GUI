#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LTX-Video-Trainer 模型下载脚本
作者: Cascade
日期: 2025-05-17
"""

import os
import sys
import shutil
from pathlib import Path
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import huggingface_hub

# 初始化Rich控制台
console = Console(highlight=False)

# 定义模型信息
MODELS = {
    "LTX-Video-0.9.7-diffusers": {
        "repo_id": "a-r-r-o-w/LTX-Video-0.9.7-diffusers",
        "local_dir": "models/LTX-Video-0.9.7-diffusers",
        "required": True
    },
    "LLaVA-NeXT-Video-7B-hf": {
        "repo_id": "llava-hf/LLaVA-NeXT-Video-7B-hf",
        "local_dir": None,  # 将使用系统默认路径
        "required": False
    },
    "T5-base": {
        "repo_id": "google-t5/t5-base",
        "local_dir": "models/t5-base",
        "required": False
    }
}

def get_user_home():
    """获取用户主目录"""
    return str(Path.home())

def download_model(model_name):
    """下载指定的模型"""
    model_info = MODELS[model_name]
    repo_id = model_info["repo_id"]
    local_dir = model_info["local_dir"]
    required = model_info["required"]
    
    # 如果是LLaVA-NeXT模型，使用系统默认目录
    if local_dir is None:
        # 让huggingface_hub使用默认缓存目录
        cache_dir = None
        console.print(f"[yellow]模型 {model_name} 将下载到系统缓存目录[/yellow]")
    else:
        # 使用指定的本地目录
        cache_dir = local_dir
        # 确保目录存在
        os.makedirs(cache_dir, exist_ok=True)
        console.print(f"[green]模型 {model_name} 将下载到 {os.path.abspath(cache_dir)}[/green]")
    
    # 开始下载
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold green]{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            f"[cyan]正在下载 {model_name}...",
            total=None,
            status="初始化中"
        )
        
        try:
            # 使用huggingface_hub下载模型
            snapshot_dir = huggingface_hub.snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_dir=cache_dir,
                local_dir_use_symlinks=False,
            )
            progress.update(task, status="下载完成")
            console.print(f"[bold green]✓ 模型 {model_name} 已成功下载到: {snapshot_dir}[/bold green]")
            return True
        except Exception as e:
            progress.update(task, status="下载失败")
            console.print(f"[bold red]× 模型 {model_name} 下载失败: {str(e)}[/bold red]")
            if required:
                console.print("[bold red]这是必需的模型，请手动下载！[/bold red]")
            return False

def main():
    """主函数"""
    console.print("[bold cyan]===== LTX-Video-Trainer 模型下载工具 =====[/bold cyan]")
    console.print("此脚本将帮助您下载训练所需的所有模型文件\n")
    
    # 确认项目根目录
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # 显示下载选项
    console.print("[bold yellow]可下载的模型:[/bold yellow]")
    for i, (name, info) in enumerate(MODELS.items(), 1):
        status = "必需" if info["required"] else "可选"
        console.print(f"  {i}. {name} [{status}]")
    
    console.print("\n[bold yellow]请选择操作:[/bold yellow]")
    console.print("  1. 下载所有模型")
    console.print("  2. 仅下载必需模型")
    console.print("  3. 选择特定模型下载")
    console.print("  0. 退出")
    
    choice = input("\n请输入选项编号 (默认:1): ").strip() or "1"
    
    if choice == "0":
        console.print("[yellow]退出程序[/yellow]")
        return
    
    models_to_download = []
    
    if choice == "1":
        # 下载所有模型
        models_to_download = list(MODELS.keys())
    elif choice == "2":
        # 仅下载必需模型
        models_to_download = [name for name, info in MODELS.items() if info["required"]]
    elif choice == "3":
        # 选择特定模型
        console.print("\n[bold yellow]请选择要下载的模型 (多个模型用逗号分隔):[/bold yellow]")
        for i, name in enumerate(MODELS.keys(), 1):
            console.print(f"  {i}. {name}")
        
        selections = input("\n请输入模型编号 (例如: 1,3): ").strip()
        try:
            indices = [int(idx.strip()) for idx in selections.split(",") if idx.strip()]
            models_to_download = [list(MODELS.keys())[i-1] for i in indices if 1 <= i <= len(MODELS)]
        except ValueError:
            console.print("[bold red]输入无效，退出程序[/bold red]")
            return
    
    if not models_to_download:
        console.print("[bold red]未选择任何模型，退出程序[/bold red]")
        return
    
    # 开始下载
    console.print(f"\n[bold green]开始下载 {len(models_to_download)} 个模型...[/bold green]")
    
    success_count = 0
    for model_name in models_to_download:
        if download_model(model_name):
            success_count += 1
    
    # 显示下载结果
    console.print(f"\n[bold {'green' if success_count == len(models_to_download) else 'yellow'}]下载完成: {success_count}/{len(models_to_download)} 个模型成功下载[/bold]")
    
    # 等待用户确认
    input("\n按回车键继续...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]下载被用户中断[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]发生错误: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
    finally:
        console.print("[cyan]程序结束[/cyan]")
