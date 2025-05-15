#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专门用于修复resolution_bucket格式的包装器
确保参数符合WxHxF格式(宽x高x帧数)
"""

import os
import sys
import subprocess
from pathlib import Path
from rich.console import Console

# 初始化控制台输出
console = Console()

def fix_resolution_format(args):
    """
    修复resolution_buckets参数格式
    确保符合WxHxF格式，缺少帧数时添加默认帧数
    """
    fixed_args = []
    
    i = 0
    while i < len(args):
        arg = args[i]
        
        if arg == "--resolution-buckets" and i + 1 < len(args):
            # 获取resolution_buckets参数值
            resolution = args[i + 1]
            
            # 检查是否需要添加帧数
            if "x" in resolution:
                x_count = resolution.count("x")
                if x_count == 1:  # 只有WxH格式，需要添加帧数
                    # 添加默认帧数49(8*6+1)
                    resolution = f"{resolution}x49"
                    print(f"修复分辨率格式: 添加默认帧数 -> {resolution}")
            
            # 添加修复后的参数
            fixed_args.append(arg)
            fixed_args.append(resolution)
            i += 2  # 跳过参数值
        else:
            fixed_args.append(arg)
            i += 1
    
    return fixed_args

def filter_unsupported_args(args):
    """
    过滤掉不支持的参数，防止传递给preprocess_dataset.py后报错
    """
    # 定义已知预处理脚本不支持但包装器可能接收的参数
    unsupported_args = ["--debug"]
    
    filtered_args = []
    skip_next = False
    
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
            
        # 检查是否是不支持的参数
        if arg in unsupported_args:
            print(f"过滤掉不支持的参数: {arg}")
            # 如果这个参数需要值，也跳过下一个参数
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                skip_next = True
            continue
        
        filtered_args.append(arg)
    
    return filtered_args

def force_zero_workers(args):
    """
    强制将num_workers参数设置为0，以避免多进程序列化错误
    """
    modified_args = []
    skip_next = False
    worker_param_found = False
    
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
            
        if arg == "--num-workers":
            # 找到num-workers参数，将其值设置为0
            modified_args.append(arg)
            modified_args.append("0")
            worker_param_found = True
            # 跳过下一个参数（原始的workers数量值）
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                skip_next = True
        else:
            modified_args.append(arg)
    
    # 如果没有找到num-workers参数，添加它
    if not worker_param_found:
        modified_args.extend(["--num-workers", "0"])
    
    print("强制设置num_workers=0以避免多进程序列化错误")
    return modified_args

if __name__ == "__main__":
    # 获取命令行参数并移除脚本名称
    args = sys.argv[1:]
    
    # 确保有足够的参数
    if len(args) < 2:
        print("错误: 参数不足")
        sys.exit(1)
    
    # 查看目标数据集路径
    dataset_path = Path(args[0]) if args else None
    if dataset_path:
        console.print(f"\n[bold blue]===== 预处理前调试信息 =====[/]")
        console.print(f"[cyan]目标数据集路径: {dataset_path}[/]")
        
        # 检查需要的文件是否存在
        media_path_file = dataset_path / "media_path.txt"
        caption_file = dataset_path / "caption.txt"
        
        missing_files = []
        if not media_path_file.exists():
            missing_files.append("media_path.txt")
        if not caption_file.exists():
            missing_files.append("caption.txt")
        
        if missing_files:
            console.print(f"[bold red]错误: 以下必需文件在 {dataset_path} 中不存在:[/]")
            for file in missing_files:
                console.print(f"[red]- {file}[/]")
            
            # 检查是否有视频文件
            video_files = list(dataset_path.glob("*.mp4")) + list(dataset_path.glob("*.avi")) + list(dataset_path.glob("*.mov"))
            if not video_files:
                console.print(f"[bold red]警告: 在数据集目录中没有找到视频文件！[/]")
                console.print(f"请确保视频文件已经被复制到 {dataset_path}。")
            else:
                console.print(f"[yellow]发现 {len(video_files)} 个视频文件在数据集目录中，但是缺少必需的索引文件。[/]")
                console.print(f"如果需要自动创建 media_path.txt，请继续，本脚本会尝试生成。")
        
        # 检查media_path.txt文件内容
        if media_path_file.exists():
            console.print(f"[green]找到media_path.txt文件: {media_path_file}[/]")
            # 输出文件内容的前几行
            try:
                with open(media_path_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:3]  # 只读取前3行
                    console.print(f"[yellow]media_path.txt文件内容示例: {lines}[/]")
                    
                    # 检查文件回车换行符问题
                    for i, line in enumerate(lines):
                        if line.strip() and not line.endswith('\n'):
                            console.print(f"[red]警告: media_path.txt第{i+1}行没有换行符[/]")
                    
                    # 检查视频文件是否存在
                    for i, line in enumerate(lines):
                        video_file = dataset_path / line.strip()
                        if video_file.exists():
                            console.print(f"[green]视频文件存在: {video_file}[/]")
                        else:
                            console.print(f"[red]警告: 视频文件不存在: {video_file}[/]")
            except Exception as e:
                console.print(f"[red]警告: 读取media_path.txt失败: {str(e)}[/]")
        
        # 如果media_path.txt不存在或为空，尝试基于目录内容创建
        if not media_path_file.exists() or media_path_file.stat().st_size == 0:
            console.print("[yellow]尝试创建media_path.txt文件...[/]")
            try:
                # 在目录中查找视频文件
                video_files = []
                for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                    video_files.extend([f.name for f in dataset_path.glob(f"*{ext}")])
                
                if video_files:
                    with open(media_path_file, 'w', encoding='utf-8') as f:
                        for video_file in video_files:
                            f.write(f"{video_file}\n")
                    console.print(f"[green]创建了media_path.txt文件，包含{len(video_files)}个视频文件[/]")
                else:
                    console.print("[red]没有找到视频文件，无法创建media_path.txt[/]")
            except Exception as e:
                console.print(f"[red]创建media_path.txt时出错: {str(e)}[/]")
        
        # 检查caption.txt文件
        if caption_file.exists():
            console.print(f"[green]找到caption.txt文件: {caption_file}[/]")
            try:
                with open(caption_file, 'r', encoding='utf-8') as f:
                    captions = f.read().strip().splitlines()
                console.print(f"[green]caption.txt包含{len(captions)}条描述[/]")
            except Exception as e:
                console.print(f"[red]读取caption.txt时出错: {str(e)}[/]")
        
        console.print("[bold blue]===== 调试信息结束 =====\n[/]")
    
    # 过滤掉不支持的参数
    args = filter_unsupported_args(args)
    
    # 修复分辨率格式
    fixed_args = fix_resolution_format(args)
    
    # 强制设置num_workers=0
    fixed_args = force_zero_workers(fixed_args)
    
    # 构建调用预处理脚本的命令
    script_dir = Path(__file__).parent.absolute()
    preprocess_script = script_dir / "preprocess_dataset.py"
    
    cmd = [sys.executable, str(preprocess_script)] + fixed_args
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    # 输出即将执行的命令
    console.print(f"[bold green]执行预处理命令: {' '.join(cmd)}[/]")
    
    try:
        # 启动子进程，继承标准输入输出
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            # 重要：不要设置encoding，让子进程直接继承标准输入输出
            # 解决Windows中文环境下的编码问题
            bufsize=1,
            universal_newlines=True
        )
        
        # 等待进程完成并获取返回码
        return_code = process.wait()
        
        if return_code != 0:
            console.print(f"[bold red]预处理脚本执行失败，返回代码: {return_code}[/]")
        else:
            console.print(f"[bold green]预处理脚本执行成功！[/]")
        
        # 退出时使用相同的返回码
        sys.exit(return_code)
    except Exception as e:
        console.print(f"[bold red]运行预处理脚本时发生错误: {str(e)}[/]")
        sys.exit(1)
