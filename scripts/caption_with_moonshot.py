#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Moonshot API为视频创建标题的脚本
这个脚本将替代原始的标注脚本，使用Moonshot视觉API而不是本地模型
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
import requests
from PIL import Image
import cv2
from rich.console import Console
from rich.progress import (
    BarColumn, 
    MofNCompleteColumn, 
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    TimeElapsedColumn, 
    TimeRemainingColumn
)

# 设置API配置
API_KEY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api_key.txt")
MOONSHOT_API_ENDPOINT = "https://api.moonshot.cn/v1/chat/completions"

console = Console()

def read_api_key():
    """从文件中读取API密钥"""
    try:
        if os.path.exists(API_KEY_FILE):
            with open(API_KEY_FILE, "r") as f:
                return f.read().strip()
        else:
            console.print("[bold red]未找到API密钥文件[/]")
            console.print(f"请创建文件：{API_KEY_FILE}，并在其中放入您的Moonshot API密钥")
            sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]读取API密钥时出错：{str(e)}[/]")
        sys.exit(1)

def extract_frames(video_path, num_frames=3):
    """从视频中提取多个关键帧"""
    try:
        # 打开视频文件
        video = cv2.VideoCapture(str(video_path))
        
        # 获取视频总帧数和帧率
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        if total_frames <= 0:
            console.print(f"[bold yellow]警告：视频 {video_path} 似乎没有帧[/]")
            return None
        
        console.print(f"[bold blue]视频信息: 总帧数={total_frames}, 帧率={fps:.2f}fps, 时长={duration:.2f}秒[/]")
        
        # 计算要提取的帧的位置
        frame_positions = []
        if num_frames == 1:
            # 只取中间帧
            frame_positions = [total_frames // 2]
        else:
            # 在视频中均匀分布提取帧
            for i in range(num_frames):
                pos = int((i / (num_frames - 1)) * (total_frames - 1)) if num_frames > 1 else total_frames // 2
                frame_positions.append(pos)
        
        # 创建临时文件路径
        temp_dir = os.path.join(os.path.dirname(video_path), ".temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 提取所有帧
        frame_paths = []
        for i, pos in enumerate(frame_positions):
            video.set(cv2.CAP_PROP_POS_FRAMES, pos)
            success, frame = video.read()
            
            if not success:
                console.print(f"[bold yellow]警告：无法从视频 {video_path} 读取帧 {pos}[/]")
                continue
            
            # 将BGR转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL图像
            pil_image = Image.fromarray(frame_rgb)
            
            # 保存为临时JPEG文件
            frame_type = "start" if i == 0 else "middle" if i == 1 else "end"
            temp_image_path = os.path.join(temp_dir, f"{Path(video_path).stem}_{frame_type}_frame.jpg")
            pil_image.save(temp_image_path)
            frame_paths.append(temp_image_path)
        
        video.release()
        
        if not frame_paths:
            console.print(f"[bold red]无法从视频 {video_path} 提取任何帧[/]")
            return None
        
        console.print(f"[bold green]成功从视频提取了 {len(frame_paths)} 个关键帧[/]")
        return frame_paths
    except Exception as e:
        console.print(f"[bold red]从视频 {video_path} 提取帧时出错：{str(e)}[/]")
        return None

def encode_image_base64(image_path):
    """将图像编码为base64字符串"""
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_video_caption(api_key, frame_paths):
    """使用Moonshot API为视频帧生成标题"""
    try:
        # 检查API密钥
        console.print(f"[bold green]检查API密钥有效性[/]")
        if not api_key or len(api_key) < 20:
            console.print("[bold red]API密钥无效或太短![/]")
            return "API key invalid or too short"
        
        console.print(f"[bold cyan]API密钥: {api_key[:5]}...{api_key[-5:]} (长度: {len(api_key)})[/]")
        
        # 用base64编码所有图像
        base64_images = []
        video_name = Path(frame_paths[0]).stem.split('_')[0] if frame_paths else "unknown"
        
        for image_path in frame_paths:
            console.print(f"[bold blue]处理图像: {image_path}[/]")
            base64_image = encode_image_base64(image_path)
            image_size_kb = len(base64_image) / 1024
            console.print(f"[bold blue]图像大小: {image_size_kb:.2f} KB[/]")
            base64_images.append(base64_image)
        
        # 创建API请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 使用视觉模型
        console.print("[bold green]尝试Moonshot视觉模型[/]")
        models_to_try = [
            "moonshot-v1-8k-vision-preview",
            "moonshot-v1-32k-vision-preview"
        ]
        
        # 尝试视觉模型
        for model in models_to_try:
            console.print(f"[bold cyan]尝试模型: {model}[/]")
            
            # 构建多模态请求 - 只针对视觉模型
            message_content = []
            
            # 添加所有帧到消息中
            for i, base64_img in enumerate(base64_images):
                frame_type = "start" if i == 0 else "middle" if i == 1 else "end"
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"
                    }
                })
            
            # 添加文本提示
            message_content.append({
                "type": "text",
                "text": "Create a detailed English caption for these video frames. Describe what is happening in the video, including all important subjects, actions, and visual elements. Provide context about the setting and atmosphere. Aim for a comprehensive description in 2-3 sentences."
            })
            
            # 完整请求数据
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional video content describer. Create detailed, descriptive English captions for videos. Your descriptions should be comprehensive, covering all important visual elements, actions, and context in 2-3 sentences."
                    },
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 200
            }
            
            # 只用于显示的调试信息 - 不影响实际发送的请求
            payload_debug = json.loads(json.dumps(payload))
            if isinstance(payload_debug.get("messages", [{}])[1].get("content", ""), list):
                for item in payload_debug["messages"][1]["content"]:
                    if item.get("type") == "image_url":
                        item["image_url"]["url"] = "[BASE64_IMAGE_DATA_REMOVED]"
            
            # 显示请求信息（但不改变原始请求）
            console.print(f"[bold yellow]发送请求（{model}）到: {MOONSHOT_API_ENDPOINT}[/]")
            console.print(f"[bold blue]请求负载类型: 多模态请求[/]")
            if len(json.dumps(payload_debug)) > 1000:
                console.print(f"[dim]调试请求负载预览: {json.dumps(payload_debug, ensure_ascii=False)[:1000]}...[/]")
            else:
                console.print(f"[dim]调试请求负载预览: {json.dumps(payload_debug, ensure_ascii=False)}[/]")
            
            # 发送请求
            console.print("[bold green]发送请求到Moonshot API...[/]")
            response = requests.post(MOONSHOT_API_ENDPOINT, headers=headers, json=payload)
            
            # 显示响应状态和头部信息
            console.print(f"[bold {'green' if response.status_code == 200 else 'red'}]状态码: {response.status_code}[/]")
            console.print(f"[bold blue]响应头关键信息:[/]")
            for key in ['content-type', 'x-request-id']:
                if key in response.headers:
                    console.print(f"  [cyan]- {key}:[/] {response.headers[key]}")
            
            # 解析响应为JSON
            try:
                if response.status_code == 200:
                    result = response.json()
                    console.print(f"[dim]响应内容: {json.dumps(result, ensure_ascii=False, indent=2)}[/]")
                    
                    # 检查响应是否有效
                    if "choices" in result and len(result["choices"]) > 0:
                        caption = result["choices"][0]["message"]["content"].strip()
                        
                        import re
                        # 处理任何格式的API响应，将其转换为单行简洁描述
                        # 显示原始响应（仅调试用）
                        if len(caption) > 100:
                            console.print(f"[bold yellow]原始标题: {caption[:100]}...[/]")
                        else:
                            console.print(f"[bold yellow]原始标题: {caption}[/]")
                        
                        # 1. 处理多行格式，保留尽可能多的内容
                        if '\n' in caption:
                            # 将所有行提取出来
                            lines = [line.strip() for line in caption.split('\n') if line.strip()]
                            
                            # 从每行移除编号和引号
                            clean_lines = []
                            for line in lines:
                                # 移除常见的编号格式 "1. ", "2) ", "- " 等
                                line = re.sub(r'^\d+[\.\)\:]\s*', '', line)
                                line = re.sub(r'^-\s+', '', line)
                                # 移除引号
                                line = line.replace('"', '').replace('"', '').replace('\'', '')
                                if line.strip():
                                    clean_lines.append(line.strip())
                            
                            # 如果只有一行，直接使用
                            if len(clean_lines) == 1 and len(clean_lines[0]) > 10:
                                caption = clean_lines[0]
                                if not any(caption.endswith(c) for c in ['.', '!', '?']):
                                    caption += '.'
                            
                            # 如果有多行，尝试将它们组合为一个连贯的描述
                            elif len(clean_lines) > 1:
                                # 先检查是否每行都描述了不同的内容
                                if all('frame' in line.lower() or 'image' in line.lower() for line in clean_lines[:3]) or \
                                   any(line.startswith(str(i+1)) for i, line in enumerate(clean_lines[:3])):
                                    # 这是对不同帧的描述，取第一个并继续
                                    caption = clean_lines[0]
                                    if not any(caption.endswith(c) for c in ['.', '!', '?']):
                                        caption += '.'
                                else:
                                    # 这些行描述了一个连贯的场景，将它们组合起来
                                    combined_text = ' '.join(clean_lines)
                                    # 如果太长，只取两行
                                    if len(combined_text) > 150:
                                        combined_text = ' '.join(clean_lines[:2])
                                    caption = combined_text
                                    if not any(caption.endswith(c) for c in ['.', '!', '?']):
                                        caption += '.'
                        
                        # 2. 如果有编号格式，去除编号
                        caption = re.sub(r'^\d+\.\s*', '', caption)
                        
                        # 3. 去除引号
                        caption = caption.replace('"', '').replace('"', '').replace('\'', '')
                        
                        # 4. 确保是英文格式（替换非ASCII字符）
                        caption = re.sub(r'[^\x00-\x7F]+', ' ', caption)
                        
                        # 5. 处理特殊情况：空白帧或无内容
                        if any(term in caption.lower() for term in ['no visible content', 'black screen', 'cannot', 'not possible', 'empty', 'blank']):
                            caption = "A blank or black screen with no visible content."
                        
                        # 6. 确保最终结果格式良好（单行，简洁，有句点结尾）
                        caption = caption.strip()
                        if not caption.endswith('.'):
                            caption += '.'
                        
                        # 确保有最小长度但不设置上限以保留完整内容
                        if len(caption) < 10:
                            caption = f"Video content showing {video_name}."
                        
                        console.print(f"[bold green]成功! 处理后标题: {caption}[/]")
                        return caption

                    else:
                        console.print(f"[bold yellow]返回状态码200，但数据格式不正确: {result}[/]")
                else:
                    console.print(f"[bold red]错误状态码 {response.status_code}[/]")
                    try:
                        error_json = response.json()
                        console.print(f"[bold red]错误内容: {json.dumps(error_json, ensure_ascii=False)}[/]")
                        error_message = error_json.get('error', {}).get('message', '未知错误')
                        console.print(f"[bold red]错误信息: {error_message}[/]")
                    except:
                        console.print(f"[bold red]无法解析错误响应: {response.text}[/]")
            except Exception as json_err:
                console.print(f"[bold red]解析响应JSON时出错: {str(json_err)}[/]")
                console.print(f"[dim]原始响应: {response.text[:500]}...[/]" if len(response.text) > 500 else f"[dim]原始响应: {response.text}[/]")
        
        # 如果视觉模型失败，尝试一个简单的基于文本的描述
        console.print("[bold yellow]视觉API请求失败，尝试生成基本描述[/]")
        try:
            # 使用普通文本模型生成视频描述
            text_model = "moonshot-v1-8k"
            console.print(f"[bold cyan]使用纯文本模型: {text_model}[/]")
            
            # 提取视频文件名作为上下文
            video_filename = Path(frame_paths[0]).stem.split('_')[0]
            
            # 构建文本请求
            text_payload = {
                "model": text_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a video content describer. Create short, descriptive English captions."
                    },
                    {
                        "role": "user",
                        "content": f"Create a brief caption for a video with filename '{video_filename}'. The video is part of a training dataset."
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 200
            }
            
            # 发送文本请求
            text_response = requests.post(MOONSHOT_API_ENDPOINT, headers=headers, json=text_payload)
            
            if text_response.status_code == 200:
                text_result = text_response.json()
                if "choices" in text_result and len(text_result["choices"]) > 0:
                    text_caption = text_result["choices"][0]["message"]["content"].strip()
                    console.print(f"[bold green]成功生成基于文件名的描述: {text_caption}[/]")
                    return text_caption
        except Exception as text_err:
            console.print(f"[bold red]生成文本描述时出错: {str(text_err)}[/]")
        
        # 最后的备用方案
        fallback_caption = f"Video showing {video_name} content"
        console.print(f"[bold red]返回默认描述: {fallback_caption}[/]")
        return fallback_caption
    
    except Exception as e:
        console.print(f"[bold red]尝试所有API调用选项时出错：{str(e)}[/]")
        video_name = Path(frame_paths[0]).stem.split('_')[0] if frame_paths and len(frame_paths) > 0 else "unknown"
        return f"Video showing {video_name} content"

def process_videos(input_dir, output_path):
    """处理目录中的所有视频文件"""
    # 支持的视频扩展名
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    
    # 寻找所有视频文件
    video_files = []
    input_path = Path(input_dir)
    
    if input_path.is_file() and input_path.suffix.lower() in video_extensions:
        video_files.append(input_path)
    else:
        for ext in video_extensions:
            video_files.extend(list(input_path.glob(f"*{ext}")))
    
    if not video_files:
        console.print(f"[bold red]在 {input_dir} 中未找到视频文件[/]")
        return
    
    console.print(f"找到 [bold]{len(video_files)}[/] 个视频文件")
    
    # 读取API密钥
    api_key = read_api_key()
    
    # 强制使用API生成标题
    console.print("[bold cyan]将强制使用Moonshot API为所有视频生成新标题[/]")
    console.print("[bold red]忽略所有现有标题，并将强制使用API重新生成所有标题[/]")
    captions = {}
    
    # 尝试打开文件仅仅是为了验证文件是否可访问
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                console.print(f"[bold green]确认可以访问输出文件 {output_path}[/]")
        except Exception as e:
            console.print(f"[bold yellow]访问输出文件时出错: {str(e)}[/]")
    
    # 创建进度条
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"标注视频", total=len(video_files))
        
        # 处理每个视频
        for video_file in video_files:
            video_path = str(video_file)
            video_name = os.path.basename(video_path)
            
            # 更新任务描述
            progress.update(task, description=f"标注 {video_name}")
            
            # 强制重新生成所有标题，无论是否已经存在
            if video_name in captions:
                console.print(f"[bold yellow]强制重新生成标题[/]: {video_name} (当前: {captions[video_name]})")
            
            # 提取多个关键帧
            frame_paths = extract_frames(video_path, num_frames=3)
            if not frame_paths:
                console.print(f"[bold red]无法提取 {video_name} 的关键帧，跳过[/]")
                progress.update(task, advance=1)
                continue
            
            # 获取标题
            caption = get_video_caption(api_key, frame_paths)
            
            # 将标题添加到字典中
            captions[video_name] = caption
            
            # 更新进度
            progress.update(task, advance=1)
            
            # 删除临时图像文件
            try:
                for frame_path in frame_paths:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
            except Exception as e:
                console.print(f"[bold yellow]删除临时文件时出错: {str(e)}[/]")
            
            # 每处理一个视频就保存一次标题文件，避免中断时丢失数据
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(captions, f, ensure_ascii=False, indent=2)
            
            # 添加一点延迟，避免API请求过于频繁
            time.sleep(0.5)
    
    # 完成后保存标题文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    
    # 使用没有特殊字符的版本以避免编码错误
    console.print(f"[bold green]标注完成，已保存到 {output_path}[/]")
    return captions

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用Moonshot API为视频创建标题")
    parser.add_argument("input", help="输入视频文件或目录")
    parser.add_argument("--output", "-o", default=None, help="输出JSON文件路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
    args = parser.parse_args()
    
    # 确定输出路径
    input_path = Path(args.input)
    if args.output:
        output_path = args.output
    else:
        if input_path.is_file():
            output_dir = input_path.parent
        else:
            output_dir = input_path
        output_path = os.path.join(output_dir, "captions.json")
    
    console.print(f"输出将保存到 [bold blue]{output_path}[/]")
    
    # 处理视频
    process_videos(args.input, output_path)

if __name__ == "__main__":
    main()
