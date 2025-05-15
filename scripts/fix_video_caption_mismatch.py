#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频与描述不匹配修复脚本
此脚本用于检测并修复视频文件与描述文件的不匹配问题
"""

import os
import json
import shutil
import argparse
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VideoDescriptionFixer")

def check_and_fix_video_caption_mismatch(dataset_path):
    """
    检查并修复视频与描述数据的不匹配问题
    
    Args:
        dataset_path: 数据集路径
        
    Returns:
        bool: 是否进行了修复操作
    """
    # 创建临时目录
    temp_dir = os.path.join(dataset_path, 'temp_unused_videos')
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"创建临时目录用于存放未使用的视频: {temp_dir}")
    
    # 读取captions.json文件
    caption_file = os.path.join(dataset_path, 'caption', 'captions.json')
    if not os.path.exists(caption_file):
        logger.error(f"找不到caption文件: {caption_file}")
        return False
        
    try:
        with open(caption_file, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
        
        # 获取有描述的视频文件名列表
        mapped_videos = set(item['media_path'] for item in captions_data)
        logger.info(f"找到{len(mapped_videos)}个有描述的视频文件")
        
        # 获取caption目录中的所有视频文件
        caption_dir = os.path.join(dataset_path, 'caption')
        all_videos = [f for f in os.listdir(caption_dir) if f.endswith('.mp4')]
        logger.info(f"caption目录中共有{len(all_videos)}个视频文件")
        
        # 找出没有描述的视频
        unmapped_videos = [video for video in all_videos if video not in mapped_videos]
        logger.info(f"发现{len(unmapped_videos)}个没有对应描述的视频文件")
        
        # 移动没有描述的视频到临时目录
        if unmapped_videos:
            for video in unmapped_videos:
                src_path = os.path.join(caption_dir, video)
                dst_path = os.path.join(temp_dir, video)
                shutil.move(src_path, dst_path)
                logger.info(f"移动未使用的视频: {video} -> {temp_dir}")
            
            return True
        return False
    except Exception as e:
        logger.error(f"检查视频描述匹配时出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='检查并修复视频与描述的不匹配问题')
    parser.add_argument('dataset_path', help='数据集路径')
    args = parser.parse_args()
    
    if check_and_fix_video_caption_mismatch(args.dataset_path):
        logger.info("成功修复视频与描述的不匹配问题")
    else:
        logger.info("没有发现需要修复的问题，或修复失败")

if __name__ == "__main__":
    main()
