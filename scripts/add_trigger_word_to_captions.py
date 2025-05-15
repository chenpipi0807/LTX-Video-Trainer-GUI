#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将触发词添加到标注文件的每个描述前面
这样LoRA模型可以学习触发词与视频风格的关联
"""

import os
import sys
import json
import argparse
import logging
import shutil
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('add-trigger-word')

def process_txt_caption_file(caption_file, trigger_word):
    """处理TXT格式的标注文件，在每个描述前添加触发词"""
    logger.info(f"处理TXT标注文件: {caption_file}")
    
    with open(caption_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 检查是否已经添加了触发词
    for line in lines:
        if '|' in line and f"<{trigger_word}>" in line:
            logger.info(f"TXT文件已包含触发词 '<{trigger_word}>'，跳过处理")
            return 0
    
    updated_lines = []
    for line in lines:
        if '|' in line:
            file_name, caption = line.strip().split('|', 1)
            # 避免重复添加触发词
            if not caption.strip().startswith(f"<{trigger_word}>"):
                # 添加触发词到描述前面
                updated_caption = f"<{trigger_word}> {caption}"
                updated_lines.append(f"{file_name}|{updated_caption}")
            else:
                updated_lines.append(line.strip())
        else:
            updated_lines.append(line.strip())
    
    # 备份原文件 - 如果备份已存在则使用时间戳命名
    backup_file = str(caption_file) + '.bak'
    if os.path.exists(backup_file):
        import time
        backup_file = f"{str(caption_file)}.bak.{int(time.time())}"
    
    shutil.copy2(caption_file, backup_file)
    logger.info(f"原文件已备份为: {backup_file}")
    
    # 写入更新后的内容
    with open(caption_file, 'w', encoding='utf-8') as f:
        for line in updated_lines:
            f.write(f"{line}\n")
    
    logger.info(f"已更新 {len(updated_lines)} 个标注，触发词 '<{trigger_word}>' 已添加")
    return len(updated_lines)

def process_json_caption_file(caption_file, trigger_word):
    """处理JSON格式的标注文件，在每个描述前添加触发词"""
    logger.info(f"处理JSON标注文件: {caption_file}")
    
    with open(caption_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查文件格式
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        # 首先检查是否已经添加了触发词
        already_has_trigger = False
        for item in data:
            if "caption" in item and item["caption"].startswith(f"<{trigger_word}>"):
                already_has_trigger = True
                break
                
        if already_has_trigger:
            logger.info(f"JSON文件已包含触发词 '<{trigger_word}>'，跳过处理")
            return 0
            
        # 添加触发词
        count = 0
        for item in data:
            if "caption" in item:
                # 确保不重复添加触发词
                if not item["caption"].startswith(f"<{trigger_word}>"):
                    item["caption"] = f"<{trigger_word}> {item['caption']}"
                    count += 1
    else:
        logger.error("不支持的JSON格式")
        return 0
    
    # 如果没有变化，则不执行写入操作
    if count == 0:
        logger.info("未检测到需要更新的标注，跳过写入操作")
        return 0
    
    # 备份原文件 - 如果备份已存在则使用时间戳命名
    backup_file = str(caption_file) + '.bak'
    if os.path.exists(backup_file):
        import time
        backup_file = f"{str(caption_file)}.bak.{int(time.time())}"
    
    # 复制原文件作为备份
    try:
        shutil.copy2(caption_file, backup_file)
        logger.info(f"原文件已备份为: {backup_file}")
    except Exception as e:
        logger.warning(f"创建备份文件失败: {str(e)}")
    
    # 写入更新后的内容
    try:
        with open(caption_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"已更新 {count} 个标注，触发词 '<{trigger_word}>' 已添加")
        return count
    except Exception as e:
        logger.error(f"写入JSON文件时出错: {str(e)}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="将触发词添加到标注文件的每个描述前面")
    parser.add_argument("dataset_path", help="数据集路径，包含caption目录的父目录")
    parser.add_argument("--trigger-word", "-t", help="触发词，默认使用数据集目录名", default=None)
    args = parser.parse_args()
    
    # 获取数据集路径
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"数据集路径不存在: {dataset_path}")
        return 1
    
    # 决定触发词
    trigger_word = args.trigger_word
    if not trigger_word:
        # 使用数据集目录名作为触发词
        trigger_word = dataset_path.name
        logger.info(f"使用数据集目录名作为触发词: {trigger_word}")
    
    # 查找标注文件
    caption_dir = dataset_path / "caption"
    if not caption_dir.exists():
        logger.error(f"标注目录不存在: {caption_dir}")
        return 1
    
    # 处理TXT格式标注文件
    txt_caption_file = caption_dir / "caption.txt"
    if txt_caption_file.exists():
        process_txt_caption_file(txt_caption_file, trigger_word)
    
    # 处理JSON格式标注文件
    json_caption_file = caption_dir / "captions.json"
    if json_caption_file.exists():
        process_json_caption_file(json_caption_file, trigger_word)
    
    logger.info("标注文件处理完成！可以继续进行预处理和训练")
    return 0

if __name__ == "__main__":
    sys.exit(main())
