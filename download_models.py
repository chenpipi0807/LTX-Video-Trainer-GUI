#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LTX-Video-Trainer 模型下载脚本
作者: pipchen
日期: 2025-05-17
"""

import os
import sys
import json
import time
import huggingface_hub
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# 初始化Rich控制台
console = Console(highlight=False)

# 镜像地址
MIRRORS = {
    "huggingface": "https://huggingface.co",
    "aliyun": "https://hf-mirror.com"  # 阿里云镜像
}

# 定义模型信息
MODELS = {
    "LTX-Video-0.9.7-diffusers": {
        "repo_id": "a-r-r-o-w/LTX-Video-0.9.7-diffusers",
        "local_dir": "models/LTX-Video-0.9.7-diffusers",
        "required": True
    },
    "LTX-Video-0.9.8-13B-distilled": {
        "repo_id": "linoyts/LTX-Video-0.9.8-13B-distilled",
        "local_dir": "models/LTX-Video-0.9.8-13B-distilled",
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

def get_model_files(repo_id, endpoint_url):
    """获取模型仓库中的所有文件列表"""
    try:
        # 正确使用huggingface_hub API
        from huggingface_hub import HfApi
        api = HfApi(endpoint=endpoint_url)
        
        # 获取仓库信息
        repo_info = api.repo_info(repo_id=repo_id)
        
        # 获取文件列表
        files_list = api.list_repo_files(repo_id=repo_id, revision=repo_info.sha)
        return files_list
    except Exception as e:
        console.print(f"[bold red]无法获取模型文件列表: {str(e)}[/bold red]")
        return []

def download_single_file(repo_id, file_path, local_dir, endpoint_url, retry_count=3, retry_delay=5):
    """下载单个文件，支持重试
    
    Args:
        repo_id (str): 仓库ID
        file_path (str): 要下载的文件路径
        local_dir (str): 本地目录
        endpoint_url (str): API端点URL
        retry_count (int): 重试次数
        retry_delay (int): 重试间隔秒数
    """
    # 直接指定到目标目录，不创建子目录
    full_local_path = os.path.join(local_dir, file_path)
    os.makedirs(os.path.dirname(full_local_path), exist_ok=True)
    
    # 检查文件是否已存在
    if os.path.exists(full_local_path):
        # 如果文件已存在且大小>0，跳过下载
        if os.path.getsize(full_local_path) > 0:
            console.print(f"[green]文件 {file_path} 已存在，跳过[/green]")
            return True
    
    # 为了显示下载进度，直接下载到最终目标位置
    # 准备一个下载成功的标志
    download_success = False
    
    # 为了避免创建模型子目录，我们先下载到缓存然后复制
    for attempt in range(retry_count):
        try:
            if attempt > 0:
                console.print(f"[yellow]正在重试下载文件 {file_path} (尝试 {attempt+1}/{retry_count})[/yellow]")
            else:
                console.print(f"[cyan]开始下载文件 {file_path}[/cyan]")
            
            # 使用不同的缓存目录以避免冲突
            cache_folder = f".hf_cache_{attempt}"
            
            # 这里不使用progress_callback，让Hugging Face显示原生进度条
            temp_file = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                cache_dir=cache_folder,
                endpoint=endpoint_url,
                force_download=attempt > 0,  # 第一次尝试用resume，后续尝试强制重新下载
                resume_download=attempt == 0,  # 第一次尝试时使用断点续传
            )
            
            # 将文件复制到目标位置
            os.makedirs(os.path.dirname(full_local_path), exist_ok=True)
            import shutil
            shutil.copy2(temp_file, full_local_path)
            
            # 清理缓存
            try:
                shutil.rmtree(cache_folder, ignore_errors=True)
            except:
                pass
            
            download_success = True
            break
        except Exception as e:
            # 清理临时缓存
            try:
                import shutil
                shutil.rmtree(f".hf_cache_{attempt}", ignore_errors=True)
            except:
                pass
                
            if attempt < retry_count - 1:
                # 尝试识别错误类型
                error_str = str(e)
                if "IncompleteRead" in error_str or "Connection broken" in error_str:
                    retry_delay_time = retry_delay * (attempt + 1)  # 每次重试增加等待时间
                    console.print(f"[yellow]网络连接中断: {error_str}，{retry_delay_time}秒后重试...[/yellow]")
                    time.sleep(retry_delay_time)
                else:
                    console.print(f"[yellow]下载失败: {error_str}，{retry_delay}秒后重试...[/yellow]")
                    time.sleep(retry_delay)
            else:
                console.print(f"[bold red]下载文件 {file_path} 失败: {str(e)}[/bold red]")
    
    return download_success

def retry_failed_files(repo_id, failed_files, local_dir, endpoint_url):
    """重试下载失败的文件"""
    if not failed_files:
        return []
    
    still_failed = []
    console.print("\n[bold yellow]开始重试下载失败的文件...[/bold yellow]")
    
    for i, file_path in enumerate(failed_files):
        console.print(f"[cyan]重新下载文件 ({i+1}/{len(failed_files)}): {file_path}[/cyan]")
        if not download_single_file(repo_id, file_path, local_dir, endpoint_url, retry_count=2):
            still_failed.append(file_path)
        # 增加成功提示
        else:
            console.print(f"[green]重新下载成功: {file_path}[/green]")
    
    return still_failed

def scan_local_files(local_dir, files_list):
    """扫描本地目录，返回已存在的文件列表和其大小"""
    existing_files = {}
    missing_files = []
    incomplete_files = []
    
    for file_path in files_list:
        full_path = os.path.join(local_dir, file_path)
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            if file_size > 0:  # 文件存在且大小大于0
                existing_files[file_path] = file_size
            else:  # 文件存在但小于或等于0，可能下载不完整
                incomplete_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    return existing_files, missing_files, incomplete_files

def download_model(model_name, mirror="huggingface"):
    """下载指定的模型
    
    Args:
        model_name (str): 要下载的模型名称
        mirror (str): 使用的镜像源，默认为原始Hugging Face
    """
    model_info = MODELS[model_name]
    repo_id = model_info["repo_id"]
    local_dir = model_info["local_dir"]
    required = model_info["required"]
    
    # 设置镜像地址
    endpoint_url = MIRRORS.get(mirror, MIRRORS["huggingface"])
    console.print(f"[blue]使用镜像源: {mirror} ({endpoint_url})[/blue]")
    
    # 如果是LLaVA-NeXT模型，使用系统默认目录
    if local_dir is None:
        # 让huggingface_hub使用默认缓存目录
        cache_dir = None
        local_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        console.print(f"[yellow]模型 {model_name} 将下载到系统缓存目录[/yellow]")
    else:
        # 使用指定的本地目录
        # 确保目录存在
        os.makedirs(local_dir, exist_ok=True)
        console.print(f"[green]模型 {model_name} 将下载到 {os.path.abspath(local_dir)}[/green]")
    
    # 获取模型文件列表
    console.print(f"[cyan]获取 {model_name} 模型文件列表...[/cyan]")
    files_list = get_model_files(repo_id, endpoint_url)
    
    if not files_list:
        console.print(f"[bold red]无法获取 {model_name} 模型文件列表，将检查本地文件并尝试整体下载[/bold red]")
        
        # 尝试检查本地文件结构
        if os.path.exists(local_dir) and os.listdir(local_dir):
            # 检查关键文件是否存在
            key_files = [
                "model_index.json",
                "config.json",
                "diffusion_pytorch_model.safetensors",
                "scheduler/scheduler_config.json"
            ]
            
            missing_key_files = []
            for key_file in key_files:
                if not os.path.exists(os.path.join(local_dir, key_file)):
                    missing_key_files.append(key_file)
            
            if not missing_key_files:
                console.print(f"[bold green]本地目录已存在关键模型文件，跳过下载[/bold green]")
                return True
            else:
                console.print(f"[yellow]本地目录缺失 {len(missing_key_files)} 个关键文件，需要下载[/yellow]")
                for file in missing_key_files[:5]:
                    console.print(f"  - {file}")
        
        try:
            # 如果无法获取文件列表，先下载到临时目录然后移动
            temp_dir = ".hf_temp_download"
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)  # 清理在之前下载过程中可能遗留的临时目录
            os.makedirs(temp_dir, exist_ok=True)
            
            console.print(f"[cyan]开始下载 {model_name} 模型...[/cyan]")
            snapshot_dir = huggingface_hub.snapshot_download(
                repo_id=repo_id,
                cache_dir=".hf_cache",
                local_dir=temp_dir,
                local_dir_use_symlinks=False,
                endpoint=endpoint_url,
                resume_download=True  # 允许断点续传
            )
            
            # 找到下载的目录，通常是 models--author--model 格式
            author, model_name_from_repo = repo_id.split('/')
            expected_subdir = f"models--{author.replace('-', '-')}--{model_name_from_repo.replace('-', '-')}"
            
            # 确定实际下载的目录
            download_dir = None
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                if os.path.isdir(item_path) and item.startswith("models--"):
                    download_dir = item_path
                    break
            
            if download_dir:
                # 移动文件到目标目录
                import shutil
                
                # 确保目标目录存在
                os.makedirs(local_dir, exist_ok=True)
                
                # 获取下载目录中的所有文件和目录
                items_to_copy = []
                for root, dirs, files in os.walk(download_dir):
                    rel_path = os.path.relpath(root, download_dir)
                    if rel_path == ".":
                        rel_path = ""
                    
                    for file in files:
                        items_to_copy.append(os.path.join(rel_path, file))
                
                # 显示文件复制进度
                console.print(f"[cyan]正在复制 {len(items_to_copy)} 个文件到目标目录...[/cyan]")
                
                # 复制文件
                for item in items_to_copy:
                    src = os.path.join(download_dir, item)
                    dst = os.path.join(local_dir, item)
                    
                    # 创建目标目录（如果不存在）
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    
                    # 复制文件
                    shutil.copy2(src, dst)
                
                # 清理临时目录
                shutil.rmtree(temp_dir)
                
                console.print(f"[bold green]✓ 模型 {model_name} 已成功下载到: {os.path.abspath(local_dir)}[/bold green]")
                return True
            else:
                console.print(f"[bold red]无法找到下载的模型文件[/bold red]")
                return False
        except Exception as e:
            console.print(f"[bold red]× 模型 {model_name} 下载失败: {str(e)}[/bold red]")
            if required:
                console.print("[bold red]这是必需的模型，请手动下载！[/bold red]")
            return False
    
    # 过滤掉.gitattributes等不需要的文件
    files_list = [f for f in files_list if not f.endswith('.gitattributes')]
    
    # 扫描本地文件，检查哪些文件已存在
    console.print(f"[cyan]扫描本地目录，检查现有模型文件...[/cyan]")
    existing_files, missing_files, incomplete_files = scan_local_files(local_dir, files_list)
    
    # 显示扫描结果
    if existing_files:
        console.print(f"[green]找到 {len(existing_files)} 个已下载完成的文件[/green]")
    
    # 需要下载的文件 = 缺失的文件 + 不完整的文件
    files_to_download = missing_files + incomplete_files
    
    if not files_to_download:
        console.print(f"[bold green]✓ 所有文件已存在，模型 {model_name} 已成功下载到: {os.path.abspath(local_dir)}[/bold green]")
        return True
    
    # 首先下载重要的配置文件
    important_files = [f for f in files_to_download if f.endswith('.json') or f == 'model_index.json']
    other_files = [f for f in files_to_download if f not in important_files]
    
    # 按重要性排序文件
    files_to_download = important_files + other_files
    
    # 按文件逐个下载
    console.print(f"[cyan]开始下载 {model_name} 模型，共 {len(files_to_download)} 个文件[/cyan]")
    
    # 记录下载失败的文件
    failed_files = []
    
    # 逐个下载文件 - 不使用自定义进度条，保留原生进度显示
    console.print(f"[bold blue]开始下载 {len(files_to_download)} 个文件，将按顺序下载[/bold blue]")
    
    completed = 0
    for file_path in files_to_download:
        success = download_single_file(repo_id, file_path, local_dir, endpoint_url)
        if success:
            completed += 1
            console.print(f"[green]进度: {completed}/{len(files_to_download)} 完成[/green]")
        else:
            failed_files.append(file_path)
    
    # 避免Rich格式化错误，使用条件选择完整的字符串
    if not failed_files:
        console.print(f"[bold green]下载完成: {completed}/{len(files_to_download)} 文件成功[/bold green]")
    else:
        console.print(f"[bold yellow]下载完成: {completed}/{len(files_to_download)} 文件成功[/bold yellow]")
    
    # 下载完成后再检查一次文件是否存在
    if failed_files:
        # 再次检查失败的文件是否实际存在（有时文件实际下载成功但报错）
        really_failed = []
        for file_path in failed_files:
            full_path = os.path.join(local_dir, file_path)
            if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
                really_failed.append(file_path)
        
        failed_files = really_failed
    
    # 如果有失败的文件，询问是否重试
    if failed_files:
        console.print(f"\n[bold yellow]有 {len(failed_files)} 个文件下载失败[/bold yellow]")
        for i, file in enumerate(failed_files[:5]):
            console.print(f"  {i+1}. {file}")
        if len(failed_files) > 5:
            console.print(f"  ... 共 {len(failed_files)} 个文件")
        
        retry = input("\n是否重试下载失败的文件? (y/n): ").strip().lower()
        if retry == 'y':
            still_failed = retry_failed_files(repo_id, failed_files, local_dir, endpoint_url)
            if still_failed:
                console.print(f"\n[bold red]仍然有 {len(still_failed)} 个文件下载失败[/bold red]")
                for i, file in enumerate(still_failed[:5]):
                    console.print(f"  {i+1}. {file}")
                if len(still_failed) > 5:
                    console.print(f"  ... 共 {len(still_failed)} 个文件")
                
                if required:
                    console.print("[bold red]这是必需的模型，请手动下载完整模型！[/bold red]")
                return False
    
    # 如果所有文件都下载成功，或者重试后所有文件都成功
    console.print(f"[bold green]✓ 模型 {model_name} 已成功下载到: {os.path.abspath(local_dir)}[/bold green]")
    return True

def main():
    """主函数"""
    console.print("[bold cyan]===== LTX-Video-Trainer 模型下载工具 =====[/bold cyan]")
    console.print("此脚本将帮助您下载训练所需的所有模型文件\n")
    
    # 确认项目根目录
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # 选择镜像源
    console.print("[bold yellow]请选择下载镜像源:[/bold yellow]")
    console.print("  1. Hugging Face官方源 (国外用户推荐)")
    console.print("  2. 阿里云镜像 (国内用户推荐，加速下载)")
    
    mirror_choice = input("\n请选择镜像源 (默认:1): ").strip() or "1"
    mirror = "huggingface" if mirror_choice == "1" else "aliyun"
    
    # 显示下载选项
    console.print("\n[bold yellow]可下载的模型:[/bold yellow]")
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
        if download_model(model_name, mirror=mirror):
            success_count += 1
    
    # 显示下载结果 - 避免Rich格式化错误
    if success_count == len(models_to_download):
        console.print(f"\n[bold green]下载完成: {success_count}/{len(models_to_download)} 个模型成功下载[/bold green]")
    else:
        console.print(f"\n[bold yellow]下载完成: {success_count}/{len(models_to_download)} 个模型成功下载[/bold yellow]")
    
    # 等待用户确认
    input("\n按回车键继续...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]用户取消下载[/yellow]")
    except Exception as e:
        # 处理错误信息中可能包含的特殊字符，避免Rich标记解析错误
        error_msg = str(e).replace("[", "\\[").replace("]", "\\]")
        console.print(f"\n[bold red]发生错误: {error_msg}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
    finally:
        console.print("[cyan]程序结束[/cyan]")
