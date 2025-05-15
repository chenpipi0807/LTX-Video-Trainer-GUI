#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
在预处理前自动修复视频与描述不匹配的问题
可以作为独立脚本运行，也可以被其他脚本导入
"""

import os
import json
import shutil
import argparse
from pathlib import Path
import logging
from typing import Tuple, List, Set

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VideoDescriptionFixer")

def check_and_fix_video_caption_mismatch(dataset_path: str) -> Tuple[bool, int, int]:
    """
    检查并修复视频与描述数据的不匹配问题
    
    Args:
        dataset_path: 数据集路径
        
    Returns:
        Tuple[bool, int, int]: 
            - 是否进行了修复操作
            - 有描述的视频数量
            - 移动的视频数量
    """
    # 创建临时目录
    temp_dir = os.path.join(dataset_path, 'temp_unused_videos')
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"创建临时目录用于存放未使用的视频: {temp_dir}")
    
    # 读取captions.json文件
    json_files = [
        os.path.join(dataset_path, 'caption', 'captions.json'),  # 首选位置
        os.path.join(dataset_path, 'captions.json')              # 备选位置
    ]
    
    captions_data = None
    caption_file = None
    
    for json_file in json_files:
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    captions_data = json.load(f)
                caption_file = json_file
                logger.info(f"找到caption文件: {json_file}")
                break
            except Exception as e:
                logger.error(f"读取caption文件时出错: {str(e)}")
    
    if captions_data is None:
        logger.error(f"找不到有效的caption文件")
        return False, 0, 0
    
    # 获取有描述的视频文件名列表
    mapped_videos = set()
    if isinstance(captions_data, list):
        # 新格式 - 列表中的字典
        for item in captions_data:
            if isinstance(item, dict) and 'media_path' in item:
                mapped_videos.add(item['media_path'])
    elif isinstance(captions_data, dict):
        # 旧格式 - 直接使用键
        mapped_videos = set(captions_data.keys())
    
    logger.info(f"找到{len(mapped_videos)}个有描述的视频文件")
    
    # 获取caption目录中的所有视频文件
    caption_dir = os.path.join(dataset_path, 'caption')
    video_dirs = [caption_dir]  # 首选位置
    
    # 如果caption目录不存在，检查数据集根目录
    if not os.path.exists(caption_dir):
        video_dirs = [dataset_path]  # 备选位置
    
    all_videos = []
    for video_dir in video_dirs:
        if os.path.exists(video_dir):
            for f in os.listdir(video_dir):
                if f.endswith('.mp4') or f.endswith('.avi') or f.endswith('.mov') or f.endswith('.mkv'):
                    all_videos.append(f)
    
    logger.info(f"目录中共有{len(all_videos)}个视频文件")
    
    # 找出没有描述的视频
    unmapped_videos = [video for video in all_videos if video not in mapped_videos]
    logger.info(f"发现{len(unmapped_videos)}个没有对应描述的视频文件")
    
    # 移动没有描述的视频到临时目录
    moved_count = 0
    if unmapped_videos:
        for video in unmapped_videos:
            try:
                # 检查文件在哪个目录中存在
                src_path = None
                for video_dir in video_dirs:
                    temp_path = os.path.join(video_dir, video)
                    if os.path.exists(temp_path):
                        src_path = temp_path
                        break
                
                if src_path:
                    dst_path = os.path.join(temp_dir, video)
                    shutil.move(src_path, dst_path)
                    logger.info(f"移动未使用的视频: {video} -> {temp_dir}")
                    moved_count += 1
            except Exception as e:
                logger.error(f"移动视频文件时出错: {str(e)}")
        
        return True, len(mapped_videos), moved_count
    
    return False, len(mapped_videos), 0

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='检查并修复视频与描述的不匹配问题')
    parser.add_argument('dataset_path', help='数据集路径')
    args = parser.parse_args()
    
    fixed, mapped_count, moved_count = check_and_fix_video_caption_mismatch(args.dataset_path)
    
    if fixed:
        logger.info(f"成功修复视频与描述的不匹配问题: 有描述视频={mapped_count}个, 移动了{moved_count}个不匹配的视频文件")
    else:
        logger.info(f"没有发现需要修复的问题，或修复失败: 有描述视频={mapped_count}个")

if __name__ == "__main__":
    main()
