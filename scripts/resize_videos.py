#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
调整视频分辨率脚本

将输入目录中的视频文件调整为指定分辨率，并保存到输出目录。
支持保持宽高比和强制指定分辨率两种模式。
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import concurrent.futures
import cv2

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LTX-Video-Resizer')

def parse_resolution(resolution_str):
    """解析分辨率字符串 (如 '448x768')"""
    if 'x' not in resolution_str:
        raise ValueError(f"分辨率格式错误，应为 'WIDTHxHEIGHT'，得到: {resolution_str}")
    
    # 处理可能带有方向标识前缀的分辨率
    if ']' in resolution_str:
        resolution_str = resolution_str.split(']')[1].strip()
    
    width, height = map(int, resolution_str.split('x'))
    return width, height

def resize_video(input_file, output_file, target_width, target_height, keep_aspect_ratio=True):
    """调整视频分辨率
    
    使用"先按照最短边缩放然后再中心裁切"的方法处理视频，确保输出的所有视频分辨率完全一致
    
    Args:
        input_file: 输入视频文件路径
        output_file: 输出视频文件路径
        target_width: 目标宽度
        target_height: 目标高度
        keep_aspect_ratio: 是否保持宽高比，如果为False则直接拉伸到目标尺寸
    
    Returns:
        成功返回True，失败返回False
    """
    try:
        # 打开输入视频
        cap = cv2.VideoCapture(str(input_file))
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {input_file}")
            return False
        
        # 获取原始视频信息
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 确保目标宽度和高度是32的倍数，这对很多AI模型很重要
        target_width = (target_width // 32) * 32
        target_height = (target_height // 32) * 32
        
        if keep_aspect_ratio:
            # 先按照最短边缩放然后再中心裁切
            # 计算原始宽高比
            orig_aspect_ratio = orig_width / orig_height
            target_aspect_ratio = target_width / target_height
            
            if orig_aspect_ratio > target_aspect_ratio:
                # 原始视频更宽，按照高度缩放后裁切两侧
                scale_factor = target_height / orig_height
                resize_width = int(orig_width * scale_factor)
                resize_height = target_height
                
                # 中心裁切
                crop_x_offset = (resize_width - target_width) // 2
                crop_y_offset = 0
            else:
                # 原始视频更高，按照宽度缩放后裁切上下
                scale_factor = target_width / orig_width
                resize_width = target_width
                resize_height = int(orig_height * scale_factor)
                
                # 中心裁切
                crop_x_offset = 0
                crop_y_offset = (resize_height - target_height) // 2
            
            logger.info(f"视频 {input_file.name} 将先缩放到 {resize_width}x{resize_height} 然后中心裁切到 {target_width}x{target_height}")
        else:
            # 直接拉伸到目标尺寸
            resize_width = target_width
            resize_height = target_height
            crop_x_offset = 0
            crop_y_offset = 0
            logger.info(f"视频 {input_file.name} 将直接拉伸到 {target_width}x{target_height}")
        
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 设置视频编解码器和输出视频对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码
        out = cv2.VideoWriter(str(output_file), fourcc, fps, (target_width, target_height))
        
        logger.info(f"调整视频 {input_file.name} 从 {orig_width}x{orig_height} 到 {target_width}x{target_height}")
        
        # 逐帧处理
        success, frame = cap.read()
        processed_frames = 0
        
        while success:
            # 先调整帧大小
            resized_frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
            
            # 如果需要裁切，进行中心裁切
            if keep_aspect_ratio and (crop_x_offset > 0 or crop_y_offset > 0):
                # 选取中心区域
                cropped_frame = resized_frame[
                    crop_y_offset:crop_y_offset+target_height, 
                    crop_x_offset:crop_x_offset+target_width
                ]
                out.write(cropped_frame)
            else:
                # 不需要裁切
                out.write(resized_frame)
            
            # 读取下一帧
            success, frame = cap.read()
            processed_frames += 1
            
            # 每处理100帧显示一次进度
            if processed_frames % 100 == 0:
                progress = processed_frames / frame_count * 100
                logger.info(f"处理进度: {progress:.1f}% ({processed_frames}/{frame_count})")
        
        # 释放资源
        cap.release()
        out.release()
        
        logger.info(f"视频调整完成: {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"处理视频 {input_file} 时出错: {e}")
        return False

def process_directory(input_dir, output_dir, target_resolution, keep_aspect_ratio=True, extensions=None):
    """处理目录中的所有视频文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        target_resolution: 目标分辨率字符串 (如 '448x768')
        keep_aspect_ratio: 是否保持宽高比
        extensions: 视频文件扩展名列表，如果为None则使用默认值
    
    Returns:
        成功处理的视频数量
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # 解析目标分辨率
    target_width, target_height = parse_resolution(target_resolution)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有视频文件
    input_path = Path(input_dir)
    video_files = []
    for ext in extensions:
        video_files.extend(list(input_path.glob(f"*{ext}")))
    
    logger.info(f"在 {input_dir} 中找到 {len(video_files)} 个视频文件")
    
    if not video_files:
        logger.warning(f"没有找到视频文件，请检查输入目录和扩展名")
        return 0
    
    # 处理每个视频文件
    success_count = 0
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
        # 提交所有任务
        future_to_file = {}
        for video_file in video_files:
            output_file = Path(output_dir) / video_file.name
            future = executor.submit(
                resize_video, 
                video_file, 
                output_file, 
                target_width, 
                target_height, 
                keep_aspect_ratio
            )
            future_to_file[future] = video_file
        
        # 处理结果
        for future in concurrent.futures.as_completed(future_to_file):
            video_file = future_to_file[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
            except Exception as e:
                logger.error(f"处理 {video_file} 时出错: {e}")
    
    logger.info(f"成功处理了 {success_count}/{len(video_files)} 个视频文件")
    return success_count

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='调整视频分辨率工具')
    parser.add_argument('input_dir', help='输入目录路径')
    parser.add_argument('--output-dir', '-o', help='输出目录路径，默认为input_dir_resized', default=None)
    parser.add_argument('--target-size', '-s', help='目标分辨率，格式如 "448x768"', required=True)
    parser.add_argument('--keep-aspect-ratio', '-k', action='store_true', help='保持原始宽高比')
    parser.add_argument('--extensions', '-e', help='视频文件扩展名，用逗号分隔，如 "mp4,avi,mov"', default='mp4,avi,mov,mkv')
    
    args = parser.parse_args()
    
    # 解析参数
    input_dir = args.input_dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # 默认输出目录
        if input_dir.endswith('/') or input_dir.endswith('\\'):
            input_dir = input_dir[:-1]
        output_dir = f"{input_dir}_resized"
    
    extensions = [f".{ext.strip()}" for ext in args.extensions.split(',')]
    
    # 执行处理
    logger.info(f"开始处理视频: 输入={input_dir}, 输出={output_dir}, 目标分辨率={args.target_size}")
    process_directory(input_dir, output_dir, args.target_size, args.keep_aspect_ratio, extensions)
    logger.info(f"视频调整完成")

if __name__ == '__main__':
    main()
