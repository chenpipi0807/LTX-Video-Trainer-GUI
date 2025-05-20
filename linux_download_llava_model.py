#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLaVA-NeXT-Video-7B-hf 模型下载脚本 (Linux版)
专门用于下载视频标注模型到本地models目录
"""

import os
import sys
import time
import shutil
from pathlib import Path

# 临时禁用离线模式以允许下载
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

import huggingface_hub
from huggingface_hub import HfApi

# 简单的彩色输出
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'

def print_color(text, color):
    """彩色打印"""
    print(f"{color}{text}{Colors.END}")

# 镜像地址
MIRRORS = {
    "huggingface": "https://huggingface.co",
    "aliyun": "https://hf-mirror.com"  # 阿里云镜像
}

# 模型信息
MODEL_REPO_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
MODEL_DIR = "models/LLaVA-NeXT-Video-7B-hf"  # 将下载到项目的models目录

def get_model_files(repo_id, endpoint_url):
    """获取模型仓库中的所有文件列表"""
    try:
        api = HfApi(endpoint=endpoint_url)
        
        # 获取仓库信息
        print_color(f"正在获取仓库 {repo_id} 的信息...", Colors.BLUE)
        repo_info = api.repo_info(repo_id=repo_id)
        
        # 获取文件列表
        print_color(f"正在获取文件列表...", Colors.BLUE)
        files_list = api.list_repo_files(repo_id=repo_id, revision=repo_info.sha)
        print_color(f"找到 {len(files_list)} 个文件", Colors.GREEN)
        return files_list
    except Exception as e:
        print_color(f"无法获取模型文件列表: {str(e)}", Colors.RED)
        return []

def download_file(repo_id, file_path, local_dir, endpoint_url, retry_count=3):
    """下载单个文件，支持重试"""
    # 确保环境变量正确设置为在线模式
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    """下载单个文件，支持重试"""
    # 完整本地路径
    full_path = os.path.join(local_dir, file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    # 检查文件是否已存在
    if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
        print_color(f"文件已存在: {file_path}", Colors.GREEN)
        return True
    
    # 下载文件
    for attempt in range(retry_count):
        try:
            if attempt > 0:
                print_color(f"正在重试下载 {file_path} (尝试 {attempt+1}/{retry_count})", Colors.YELLOW)
            else:
                print_color(f"下载文件: {file_path}", Colors.CYAN)
            
            # 使用临时缓存目录
            cache_dir = f".hf_cache_{attempt}"
            
            # 下载文件
            temp_file = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                cache_dir=cache_dir,
                endpoint=endpoint_url,
                force_download=attempt > 0,
                resume_download=True
            )
            
            # 复制到最终位置
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            shutil.copy2(temp_file, full_path)
            
            # 清理缓存
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
            except:
                pass
                
            return True
            
        except Exception as e:
            # 清理缓存
            try:
                shutil.rmtree(f".hf_cache_{attempt}", ignore_errors=True)
            except:
                pass
                
            error_msg = str(e)
            print_color(f"下载失败: {error_msg}", Colors.YELLOW)
            
            if attempt < retry_count - 1:
                retry_delay = 3 * (attempt + 1)
                print_color(f"{retry_delay}秒后重试...", Colors.YELLOW)
                time.sleep(retry_delay)
            else:
                print_color(f"下载文件 {file_path} 失败，已达到最大重试次数", Colors.RED)
    
    return False

def scan_local_files(local_dir, files_list):
    """扫描本地目录，返回已存在和缺失的文件"""
    existing_files = []
    missing_files = []
    
    for file_path in files_list:
        full_path = os.path.join(local_dir, file_path)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    return existing_files, missing_files

def download_model(mirror="aliyun"):
    """下载LLaVA-NeXT-Video-7B-hf模型"""
    # 准备本地目录
    local_dir = os.path.abspath(MODEL_DIR)
    os.makedirs(local_dir, exist_ok=True)
    
    # 确保环境变量正确设置为在线模式
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    
    # 设置镜像
    endpoint_url = MIRRORS.get(mirror, MIRRORS["huggingface"])
    print_color(f"使用镜像: {mirror} ({endpoint_url})", Colors.BLUE)
    print_color(f"模型将下载到: {local_dir}", Colors.BLUE)
    
    # 获取文件列表
    files_list = get_model_files(MODEL_REPO_ID, endpoint_url)
    if not files_list:
        print_color("无法获取文件列表，尝试使用snapshot_download下载整个仓库", Colors.YELLOW)
        try:
            print_color("开始下载模型...", Colors.BLUE)
            temp_dir = ".hf_temp_download"
            os.makedirs(temp_dir, exist_ok=True)
            
            # 下载整个仓库
            huggingface_hub.snapshot_download(
                repo_id=MODEL_REPO_ID,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                endpoint=endpoint_url,
                resume_download=True
            )
            
            print_color(f"模型下载完成: {local_dir}", Colors.GREEN)
            return True
        except Exception as e:
            print_color(f"下载失败: {str(e)}", Colors.RED)
            return False
    
    # 过滤不需要的文件
    files_list = [f for f in files_list if not f.endswith('.gitattributes')]
    
    # 检查哪些文件已经存在
    existing_files, missing_files = scan_local_files(local_dir, files_list)
    print_color(f"找到 {len(existing_files)} 个已下载文件，需要下载 {len(missing_files)} 个文件", Colors.BLUE)
    
    if not missing_files:
        print_color("所有文件已存在，无需下载", Colors.GREEN)
        return True
    
    # 优先下载配置文件
    config_files = [f for f in missing_files if f.endswith('.json')]
    other_files = [f for f in missing_files if not f.endswith('.json')]
    files_to_download = config_files + other_files
    
    # 下载文件
    print_color(f"开始下载 {len(files_to_download)} 个文件...", Colors.BLUE)
    success_count = 0
    failed_files = []
    
    for i, file_path in enumerate(files_to_download):
        print_color(f"[{i+1}/{len(files_to_download)}] 下载中...", Colors.CYAN)
        if download_file(MODEL_REPO_ID, file_path, local_dir, endpoint_url):
            success_count += 1
        else:
            failed_files.append(file_path)
    
    # 结果统计
    if failed_files:
        print_color(f"下载完成: {success_count}/{len(files_to_download)} 个文件成功", Colors.YELLOW)
        print_color(f"有 {len(failed_files)} 个文件下载失败", Colors.RED)
        for i, file in enumerate(failed_files[:5]):
            print_color(f"  {i+1}. {file}", Colors.RED)
        if len(failed_files) > 5:
            print_color(f"  ...共 {len(failed_files)} 个文件", Colors.RED)
        
        retry = input("\n是否重试下载失败的文件? (y/n): ").strip().lower()
        if retry == 'y':
            print_color("开始重试下载失败的文件...", Colors.YELLOW)
            still_failed = []
            for i, file_path in enumerate(failed_files):
                print_color(f"[{i+1}/{len(failed_files)}] 重试下载: {file_path}", Colors.CYAN)
                if download_file(MODEL_REPO_ID, file_path, local_dir, endpoint_url, retry_count=2):
                    success_count += 1
                else:
                    still_failed.append(file_path)
            
            if still_failed:
                print_color(f"仍有 {len(still_failed)} 个文件下载失败", Colors.RED)
                return False
    
    print_color(f"模型下载完成! 成功下载 {success_count} 个文件到: {local_dir}", Colors.GREEN)
    return True

def main():
    """主函数"""
    print_color("===== LLaVA-NeXT-Video-7B-hf 模型下载工具 =====", Colors.CYAN)
    print_color("此脚本将帮助您下载视频标注模型到本地models目录\n", Colors.CYAN)
    
    # 显示环境状态
    offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
    print_color(f"原始离线模式状态: {offline_mode}", Colors.BLUE)
    print_color("已临时禁用离线模式以允许下载", Colors.YELLOW)
    
    # 创建models目录
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # 选择镜像
    print_color("请选择下载镜像源:", Colors.YELLOW)
    print("  1. Hugging Face官方源 (国外用户推荐)")
    print("  2. 阿里云镜像 (国内用户推荐，加速下载)")
    
    mirror_choice = input("\n请选择镜像源 (默认:2): ").strip() or "2"
    mirror = "huggingface" if mirror_choice == "1" else "aliyun"
    
    # 开始下载
    print_color("\n开始下载 LLaVA-NeXT-Video-7B-hf 模型...", Colors.GREEN)
    success = download_model(mirror)
    
    if success:
        # 修改captioning.py脚本中的模型路径
        src_dir = os.path.join("src", "ltxv_trainer", "captioning")
        if os.path.exists(src_dir):
            caption_files = [
                os.path.join(src_dir, f) for f in os.listdir(src_dir) 
                if f.endswith('.py') and os.path.isfile(os.path.join(src_dir, f))
            ]
            
            for file_path in caption_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查是否包含LLaVA模型路径
                    if 'llava-hf/LLaVA-NeXT-Video-7B-hf' in content:
                        print_color(f"找到模型引用文件: {file_path}", Colors.YELLOW)
                        print_color("自动修改脚本以使用本地模型路径...", Colors.YELLOW)
                        
                        # 替换模型路径
                        local_path = os.path.abspath(MODEL_DIR)
                        new_content = content.replace('llava-hf/LLaVA-NeXT-Video-7B-hf', local_path)
                        
                        # 写回文件
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                            
                        print_color(f"✓ 已更新模型路径: {file_path}", Colors.GREEN)
                        print_color(f"  将 'llava-hf/LLaVA-NeXT-Video-7B-hf' 修改为 '{local_path}'", Colors.GREEN)
                except Exception as e:
                    print_color(f"无法修改文件 {file_path}: {str(e)}", Colors.RED)
                
        print_color("\n下载成功完成！", Colors.GREEN)
        print_color("现在您可以运行训练脚本并使用本地模型进行标注", Colors.GREEN)
    else:
        print_color("\n下载未完全成功。建议使用Moonshot API进行标注。", Colors.RED)
    
    input("\n按回车键继续...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_color("\n用户取消下载", Colors.YELLOW)
    except Exception as e:
        print_color(f"\n发生错误: {str(e)}", Colors.RED)
        import traceback
        print(traceback.format_exc())
    finally:
        print_color("程序结束", Colors.CYAN)
