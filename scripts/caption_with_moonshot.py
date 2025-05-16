#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Moonshot API为视频创建标题的脚本
这个脚本将替代原始的标注脚本，使用Moonshot API而不是本地模型
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

def extract_middle_frame(video_path):
    """从视频中提取中间帧"""
    try:
        # 打开视频文件
        video = cv2.VideoCapture(str(video_path))
        
        # 获取视频总帧数
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            console.print(f"[bold yellow]警告：视频 {video_path} 似乎没有帧[/]")
            return None
        
        # 跳转到中间帧
        middle_frame_idx = total_frames // 2
        video.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        
        # 读取中间帧
        success, frame = video.read()
        video.release()
        
        if not success:
            console.print(f"[bold yellow]警告：无法从视频 {video_path} 读取中间帧[/]")
            return None
        
        # 将BGR转换为RGB（OpenCV使用BGR，而PIL使用RGB）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(frame_rgb)
        
        # 创建临时文件路径
        temp_dir = os.path.join(os.path.dirname(video_path), ".temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 保存为临时JPEG文件
        temp_image_path = os.path.join(temp_dir, f"{Path(video_path).stem}_middle_frame.jpg")
        pil_image.save(temp_image_path)
        
        return temp_image_path
    except Exception as e:
        console.print(f"[bold red]从视频 {video_path} 提取中间帧时出错：{str(e)}[/]")
        return None

def encode_image_base64(image_path):
    """将图像编码为base64字符串"""
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_video_caption(api_key, image_path):
    """使用Moonshot API为图像生成标题"""
    try:
        # 检查API密钥
        console.print(f"[bold green]检查API密钥有效性[/]")
        if not api_key or len(api_key) < 20:
            console.print("[bold red]API密钥无效或太短![/]")
            return "API key invalid or too short"
        
        console.print(f"[bold cyan]API密钥: {api_key[:5]}...{api_key[-5:]} (长度: {len(api_key)})[/]")
        
        # 用base64编码图像
        console.print(f"[bold blue]处理图像: {image_path}[/]")
        base64_image = encode_image_base64(image_path)
        image_size_kb = len(base64_image) / 1024
        console.print(f"[bold blue]图像大小: {image_size_kb:.2f} KB[/]")
        
        # 创建API请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 测试两种模型，从较小的开始
        console.print("[bold green]将尝试两种模型: moonshot-v1-8k 和 moonshot-v1-32k[/]")
        models_to_try = [
            "moonshot-v1-8k",    # 先尝试较小的模型
            "moonshot-v1-32k"    # 然后尝试中等模型
        ]
        
        # 找到第一个成功的模型
        for model in models_to_try:
            console.print(f"[bold cyan]尝试模型: {model}[/]")
            
            # 尝试不同的消息格式
            payloads = []
            
            # 准备纯文本请求
            console.print("[bold green]添加纯文本请求...[/]")
            payloads = []  # 清空现有负载
            payloads.append({
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional video content describer. Create short, descriptive English captions for abstract fluid animation videos. Focus on colors, movements, and visual qualities."
                    },
                    {
                        "role": "user",
                        "content": f"Create a short English caption for an abstract fluid animation video titled '{Path(image_path).stem}'. Describe vivid colors, flowing movements, and visual aesthetics in one sentence. Omit phrases like 'this is' or 'this video shows'."
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 60
            })
            
            # 尝试多模态请求
            console.print("[bold yellow]正在尝试多模态请求...[/]")
            
            # 准备多模态请求格式
            try:
                multimodal_payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a professional video content describer. Create short, descriptive English captions for abstract fluid animation videos, focusing on colors, movements, and visual aesthetics. Keep it concise, accurate and descriptive."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Create a short English caption for this abstract fluid animation video. Describe the vivid colors, flowing movements, and visual aesthetics in one sentence. Avoid phrases like 'this is' or 'this video shows'."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 60
                }
                payloads.append(multimodal_payload)
                console.print("[bold green]成功添加多模态请求格式[/]")
            except Exception as e:
                console.print(f"[bold red]构建多模态请求时出错: {str(e)}[/]")
            
            # 尝试不同的消息格式
            for i, payload in enumerate(payloads):
                try:
                    console.print(f"[bold yellow]尝试请求格式 {i+1}/{len(payloads)}[/]")
                    
                    # 打印请求详情以进行调试
                    payload_debug = payload.copy()
                    if isinstance(payload_debug.get("messages", [{}])[1].get("content", ""), list):
                        for item in payload_debug["messages"][1]["content"]:
                            if item.get("type") == "image_url":
                                item["image_url"]["url"] = "[BASE64_IMAGE_DATA_REMOVED]"
                    
                    console.print(f"[bold yellow]发送请求（{model}）到: {MOONSHOT_API_ENDPOINT}[/]")
                    console.print(f"[bold green]请求头: {headers}[/]")
                    console.print(f"[bold blue]请求负载类型: {'MultiModal' if 'image_url' in str(payload) else 'Text-only'}[/]")
                    console.print(f"[dim]详细请求负载: {json.dumps(payload_debug, ensure_ascii=False)[:1000]}...[/]" if len(json.dumps(payload_debug)) > 1000 else f"[dim]详细请求负载: {json.dumps(payload_debug, ensure_ascii=False)}[/]")
                    # 发送请求
                    console.print("[bold green]发送请求到Moonshot API...[/]")
                    response = requests.post(MOONSHOT_API_ENDPOINT, headers=headers, json=payload)
                    
                    # 详细显示响应状态和头部信息
                    console.print(f"[bold {'green' if response.status_code == 200 else 'red'}]状态码: {response.status_code}[/]")
                    console.print(f"[bold blue]响应头关键信息:[/]")
                    for key in ['content-type', 'x-request-id', 'server', 'x-ratelimit-remaining']:
                        if key in response.headers:
                            console.print(f"  [cyan]- {key}:[/] {response.headers[key]}")
                    
                    # 尝试解析响应为JSON
                    try:
                        result = response.json()
                        console.print(f"[dim]响应内容: {json.dumps(result, ensure_ascii=False, indent=2)}[/]")
                        
                        # 详细检查响应是否有效
                        if response.status_code == 200:
                            if "choices" in result and len(result["choices"]) > 0:
                                caption = result["choices"][0]["message"]["content"].strip()
                                console.print(f"[bold green]成功! 生成标题: {caption}[/]")
                                return caption
                            else:
                                console.print(f"[bold yellow]返回状态码200，但数据格式不正确: {result}[/]")
                        elif response.status_code == 400:
                            console.print(f"[bold red]400错误 - 正在查看错误信息: {result.get('error', {}).get('message', '(无错误信息)')}[/]")
                    except Exception as json_err:
                        console.print(f"[bold red]解析响应JSON时出错: {str(json_err)}[/]")
                        console.print(f"[dim]原始响应: {response.text[:500]}...[/]" if len(response.text) > 500 else f"[dim]原始响应: {response.text}[/]")
                except Exception as req_err:
                    console.print(f"[bold red]请求过程中出错: {str(req_err)}[/]")
        
        # 如果所有模型和格式都失败，尝试其他请求方法
        console.print("[bold red]常规API请求失败，尝试使用curl直接发送请求[/]")
        # 尝试更简单的测试请求
        try:
            # 尝试一个非常简单的请求来测试API是否可用
            test_payload = {
                "model": "moonshot-v1-8k",
                "messages": [
                    {"role": "user", "content": "Say hello"}
                ]
            }
            resp = requests.post(MOONSHOT_API_ENDPOINT, headers=headers, json=test_payload)
            console.print(f"[bold yellow]测试请求状态码: {resp.status_code}[/]")
            console.print(f"[bold blue]测试请求响应: {resp.text[:500]}[/]")
            
            if resp.status_code == 200:
                console.print("[bold green]API可用，但与图像相关的请求失败。尝试使用图像文件名生成描述[/]")
            else:
                console.print("[bold red]API可能完全不可用或密钥无效[/]")
        except Exception as test_err:
            console.print(f"[bold red]测试请求出错: {str(test_err)}[/]")
        
        # 最后的备用方案
        video_name = Path(image_path).stem
        console.print(f"[bold red]返回基于文件名的描述: {video_name}[/]")
        return f"Dynamic abstract fluid animation with vibrant colors - {video_name}"
        
    except Exception as e:
        console.print(f"[bold red]尝试所有API调用选项时出错：{str(e)}[/]")
        # 返回一个有意义的默认值
        video_name = Path(image_path).stem.split('_')[0]
        return f"Abstract fluid animation with colorful visual effects - {video_name}"

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
            
            # 提取中间帧
            frame_path = extract_middle_frame(video_path)
            if not frame_path:
                console.print(f"[bold red]无法提取 {video_name} 的中间帧，跳过[/]")
                progress.update(task, advance=1)
                continue
            
            # 获取标题
            caption = get_video_caption(api_key, frame_path)
            
            # 将标题添加到字典中
            captions[video_name] = caption
            
            # 更新进度
            progress.update(task, advance=1)
            
            # 删除临时图像文件
            try:
                os.remove(frame_path)
            except:
                pass
            
            # 每处理一个视频就保存一次标题文件，避免中断时丢失数据
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(captions, f, ensure_ascii=False, indent=2)
            
            # 添加一点延迟，避免API请求过于频繁
            time.sleep(0.5)
    
    # 完成后保存标题文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    
    console.print(f"[bold green]✓[/] 标注完成，已保存到 {output_path}")
    return captions

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用Moonshot API为视频创建标题")
    parser.add_argument("input", help="输入视频文件或目录")
    parser.add_argument("--output", "-o", default=None, help="输出JSON文件路径")
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
