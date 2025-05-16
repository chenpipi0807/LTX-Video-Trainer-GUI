#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频帧提取脚本

从输入目录中的视频文件提取指定数量的帧，并保存到输出目录。
支持均匀提取和场景感知提取两种模式。
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import concurrent.futures
import cv2
import numpy as np
import json

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LTX-Frame-Extractor')

def extract_frames_uniform(video_path, output_dir, num_frames, file_format="frame_{:04d}.jpg"):
    """均匀提取视频帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录路径
        num_frames: 要提取的帧数
        file_format: 输出文件名格式
    
    Returns:
        成功提取的帧数量，失败返回0
    """
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return 0
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"视频信息: {video_path.name}, 总帧数: {total_frames}, FPS: {fps:.2f}, 时长: {duration:.2f}秒")
        
        if total_frames <= 0:
            logger.error(f"无法获取视频帧数: {video_path}")
            cap.release()
            return 0
        
        # 创建输出目录
        video_output_dir = Path(output_dir) / video_path.stem
        os.makedirs(video_output_dir, exist_ok=True)
        
        # 计算提取间隔
        step = total_frames / num_frames
        
        # 创建帧索引到文件名的映射
        frame_mapping = {}
        
        # 提取帧
        success_count = 0
        for i in range(num_frames):
            # 计算当前帧索引
            frame_idx = min(int(i * step), total_frames - 1)
            
            # 设置读取位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # 读取帧
            success, frame = cap.read()
            if not success:
                logger.warning(f"读取第 {frame_idx} 帧失败")
                continue
            
            # 保存帧
            output_file = str(video_output_dir / file_format.format(i+1))
            cv2.imwrite(output_file, frame)
            
            # 添加到映射
            frame_mapping[i+1] = {
                "file": output_file,
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps if fps > 0 else 0
            }
            
            success_count += 1
        
        # 保存帧映射信息
        mapping_file = video_output_dir / "frame_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump({
                "video_file": str(video_path),
                "total_frames": total_frames,
                "fps": fps,
                "duration": duration,
                "extracted_frames": success_count,
                "frames": frame_mapping
            }, f, indent=2)
        
        # 关闭视频
        cap.release()
        
        logger.info(f"成功从 {video_path.name} 提取了 {success_count} 帧")
        return success_count
    
    except Exception as e:
        logger.error(f"处理视频 {video_path} 时出错: {e}")
        return 0

def extract_frames_scene_aware(video_path, output_dir, min_frames=25, threshold=30, file_format="frame_{:04d}.jpg"):
    """场景感知帧提取
    
    使用场景变化检测提取关键帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录路径
        min_frames: 最少提取的帧数
        threshold: 场景检测阈值
        file_format: 输出文件名格式
    
    Returns:
        成功提取的帧数量，失败返回0
    """
    try:
        # 导入scenedetect库
        try:
            from scenedetect import ContentDetector, SceneManager
            from scenedetect import open_video
            from scenedetect.scene_manager import save_images
        except ImportError:
            logger.error("缺少必要的库: scenedetect. 请使用 'pip install scenedetect[opencv]' 安装")
            return 0
        
        # 创建输出目录
        video_output_dir = Path(output_dir) / video_path.stem
        os.makedirs(video_output_dir, exist_ok=True)
        
        # 打开视频
        video = open_video(str(video_path))
        
        # 创建场景管理器和检测器
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        
        # 检测场景
        scene_manager.detect_scenes(video)
        
        # 获取场景列表
        scene_list = scene_manager.get_scene_list()
        
        # 如果场景数少于最小帧数，则使用均匀提取
        if len(scene_list) < min_frames:
            logger.info(f"检测到的场景数 ({len(scene_list)}) 少于最小帧数 ({min_frames})，使用均匀提取")
            return extract_frames_uniform(video_path, output_dir, min_frames, file_format)
        
        # 保存场景关键帧
        num_saved = save_images(
            scene_list, 
            video, 
            str(video_output_dir),
            num_images=1,  # 每个场景保存1帧
            image_name_template=file_format[:-4],  # 移除扩展名
            image_extension="jpg"
        )
        
        logger.info(f"成功从 {video_path.name} 提取了 {num_saved} 个场景关键帧")
        return num_saved
    
    except Exception as e:
        logger.error(f"处理视频 {video_path} 时出错: {e}")
        return 0

def process_directory(input_dir, output_dir, num_frames=25, mode="uniform", extensions=None):
    """处理目录中的所有视频文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        num_frames: 每个视频要提取的帧数
        mode: 提取模式，"uniform"(均匀) 或 "scene"(场景感知)
        extensions: 视频文件扩展名列表
    
    Returns:
        成功处理的视频数量
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
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
    total_frames = 0
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
        # 提交所有任务
        future_to_file = {}
        for video_file in video_files:
            if mode == "uniform":
                future = executor.submit(
                    extract_frames_uniform,
                    video_file,
                    output_dir,
                    num_frames
                )
            else:  # scene mode
                future = executor.submit(
                    extract_frames_scene_aware,
                    video_file,
                    output_dir,
                    num_frames
                )
            future_to_file[future] = video_file
        
        # 处理结果
        for future in concurrent.futures.as_completed(future_to_file):
            video_file = future_to_file[future]
            try:
                frames = future.result()
                if frames > 0:
                    success_count += 1
                    total_frames += frames
            except Exception as e:
                logger.error(f"处理 {video_file} 时出错: {e}")
    
    logger.info(f"成功处理了 {success_count}/{len(video_files)} 个视频文件，共提取 {total_frames} 帧")
    return success_count

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='视频帧提取工具')
    parser.add_argument('input_dir', help='输入目录路径')
    parser.add_argument('--output-dir', '-o', help='输出目录路径，默认为input_dir_frames', default=None)
    parser.add_argument('--num-frames', '-n', type=int, help='每个视频要提取的帧数，默认为25', default=25)
    parser.add_argument('--mode', '-m', choices=['uniform', 'scene'], help='提取模式：uniform(均匀) 或 scene(场景感知)', default='uniform')
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
        output_dir = f"{input_dir}_frames"
    
    extensions = [f".{ext.strip()}" for ext in args.extensions.split(',')]
    
    # 执行处理
    logger.info(f"开始提取视频帧: 输入={input_dir}, 输出={output_dir}, 模式={args.mode}, 帧数={args.num_frames}")
    process_directory(input_dir, output_dir, args.num_frames, args.mode, extensions)
    logger.info(f"帧提取完成")

if __name__ == '__main__':
    main()
