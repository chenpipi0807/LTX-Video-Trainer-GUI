#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTX-Video-Trainer 高级训练UI组件
为专业版UI提供高级训练参数控制界面
"""

import os
import sys
import json
import gradio as gr
from enhanced_training_params import *

def create_lora_params_ui():
    """创建LoRA参数UI组件"""
    with gr.Group():
        gr.Markdown("### LoRA参数设置")
        with gr.Row():
            lora_rank = gr.Slider(
                label="LoRA秩 (Rank)", 
                minimum=1, 
                maximum=256, 
                step=4,
                value=32,
                info="值越大，适应性越强但需要更多显存和训练时间"
            )
            lora_alpha = gr.Slider(
                label="LoRA Alpha", 
                minimum=1, 
                maximum=256, 
                step=4,
                value=32,
                info="通常设置为与Rank相同的值"
            )
        
        with gr.Row():
            lora_dropout = gr.Slider(
                label="LoRA Dropout", 
                minimum=0.0, 
                maximum=0.5, 
                step=0.05,
                value=0.05,
                info="用于防止过拟合，值越大正则化越强"
            )
            
        gr.Markdown("#### 目标模块")
        target_modules = gr.CheckboxGroup(
            label="目标模块", 
            choices=["to_k", "to_q", "to_v", "to_out.0"],
            value=["to_k", "to_q", "to_v", "to_out.0"],
            info="选择要训练的模块，默认为注意力机制的所有关键部分"
        )
        
    return {
        "lora_rank": lora_rank, 
        "lora_alpha": lora_alpha, 
        "lora_dropout": lora_dropout,
        "target_modules": target_modules
    }

def create_optimization_params_ui():
    """创建优化器参数UI组件"""
    with gr.Group():
        gr.Markdown("### 优化参数设置")
        with gr.Row():
            learning_rate = gr.Slider(
                label="学习率", 
                minimum=0.00001, 
                maximum=0.001, 
                step=0.00001,
                value=0.0002,
                info="控制模型学习速度，通常在0.0001-0.0003之间"
            )
            lr_scheduler = gr.Dropdown(
                label="学习率调度器", 
                choices=LR_SCHEDULER_OPTIONS,
                value="cosine",
                info="决定学习率如何随时间变化"
            )
            
        with gr.Row():
            steps = gr.Slider(
                label="训练步数", 
                minimum=5, 
                maximum=500, 
                step=5,
                value=50,
                info="总训练迭代次数，步数越多训练时间越长"
            )
            gradient_accumulation_steps = gr.Slider(
                label="梯度累积步数", 
                minimum=1, 
                maximum=8, 
                step=1,
                value=1,
                info="用于模拟更大的批量大小，可减少显存需求"
            )
            
        with gr.Row():
            max_grad_norm = gr.Slider(
                label="梯度裁剪阈值", 
                minimum=0.1, 
                maximum=5.0, 
                step=0.1,
                value=1.0,
                info="限制梯度的最大范数，防止梯度爆炸"
            )
            
        with gr.Row():
            optimizer_type = gr.Dropdown(
                label="优化器类型", 
                choices=OPTIMIZER_OPTIONS,
                value="adamw",
                info="不同的优化算法"
            )
            enable_gradient_checkpointing = gr.Checkbox(
                label="启用梯度检查点", 
                value=True,
                info="以计算时间换取显存，减少显存使用"
            )
            
        with gr.Accordion("学习率预热设置", open=False):
            with gr.Row():
                lr_warmup_steps = gr.Slider(
                    label="预热步数", 
                    minimum=0, 
                    maximum=50, 
                    step=1,
                    value=5,
                    info="从较小学习率逐渐增加到目标学习率的步数"
                )
                lr_warmup_ratio = gr.Slider(
                    label="预热比例", 
                    minimum=0.0, 
                    maximum=0.2, 
                    step=0.01,
                    value=0.05,
                    info="预热步数占总步数的比例"
                )
        
    return {
        "learning_rate": learning_rate,
        "lr_scheduler": lr_scheduler,
        "steps": steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_grad_norm": max_grad_norm,
        "optimizer_type": optimizer_type,
        "enable_gradient_checkpointing": enable_gradient_checkpointing,
        "lr_warmup_steps": lr_warmup_steps,
        "lr_warmup_ratio": lr_warmup_ratio
    }

def create_acceleration_params_ui():
    """创建加速参数UI组件"""
    with gr.Group():
        gr.Markdown("### 加速与性能设置")
        with gr.Row():
            mixed_precision_mode = gr.Dropdown(
                label="混合精度模式", 
                choices=MIXED_PRECISION_OPTIONS,
                value="fp16",
                info="降低精度以加速训练并减少显存使用"
            )
            load_text_encoder_in_8bit = gr.Checkbox(
                label="8比特加载文本编码器", 
                value=False,
                info="使用量化加载文本编码器以节省显存"
            )
            
        with gr.Row():
            memory_efficient_attention = gr.Checkbox(
                label="内存高效注意力", 
                value=True,
                info="使用优化的注意力计算方法减少显存使用"
            )
            use_xformers = gr.Checkbox(
                label="使用xFormers", 
                value=True,
                info="使用xFormers库进行更高效的注意力计算"
            )
            
    return {
        "mixed_precision_mode": mixed_precision_mode,
        "load_text_encoder_in_8bit": load_text_encoder_in_8bit,
        "memory_efficient_attention": memory_efficient_attention,
        "use_xformers": use_xformers
    }

def create_data_params_ui():
    """创建数据处理参数UI组件"""
    with gr.Group():
        gr.Markdown("### 数据处理设置")
        with gr.Row():
            num_dataloader_workers = gr.Slider(
                label="数据加载线程数", 
                minimum=0, 
                maximum=8, 
                step=1,
                value=0,
                info="多线程数据加载，0表示使用主线程"
            )
            shuffle_batches = gr.Checkbox(
                label="随机打乱批次", 
                value=True,
                info="训练时随机打乱数据顺序"
            )
            
        with gr.Accordion("图像增强设置", open=False):
            with gr.Row():
                enable_augmentation = gr.Checkbox(
                    label="启用图像增强", 
                    value=True,
                    info="使用数据增强技术扩充训练数据"
                )
                horizontal_flip_p = gr.Slider(
                    label="水平翻转概率", 
                    minimum=0.0, 
                    maximum=1.0, 
                    step=0.1,
                    value=0.0,
                    info="随机水平翻转的概率"
                )
                
            with gr.Row():
                vertical_flip_p = gr.Slider(
                    label="垂直翻转概率", 
                    minimum=0.0, 
                    maximum=1.0, 
                    step=0.1,
                    value=0.0,
                    info="随机垂直翻转的概率"
                )
                random_crop_p = gr.Slider(
                    label="随机裁剪概率", 
                    minimum=0.0, 
                    maximum=1.0, 
                    step=0.1,
                    value=0.0,
                    info="随机裁剪的概率"
                )
                
    return {
        "num_dataloader_workers": num_dataloader_workers,
        "shuffle_batches": shuffle_batches,
        "enable_augmentation": enable_augmentation,
        "horizontal_flip_p": horizontal_flip_p,
        "vertical_flip_p": vertical_flip_p,
        "random_crop_p": random_crop_p
    }

def create_advanced_options_ui():
    """创建高级选项UI组件"""
    with gr.Group():
        gr.Markdown("### 高级训练选项")
        with gr.Row():
            first_frame_conditioning_p = gr.Slider(
                label="首帧条件概率", 
                minimum=0.0, 
                maximum=1.0, 
                step=0.1,
                value=0.5,
                info="使用第一帧作为条件的概率"
            )
            train_text_encoder = gr.Checkbox(
                label="训练文本编码器", 
                value=False,
                info="同时训练文本编码器部分"
            )
            
        with gr.Accordion("时间步采样设置", open=False):
            with gr.Row():
                timestep_sampling_mode = gr.Dropdown(
                    label="时间步采样模式", 
                    choices=TIMESTEP_SAMPLING_OPTIONS,
                    value="shifted_logit_normal",
                    info="决定如何采样去噪时间步"
                )
                
    return {
        "first_frame_conditioning_p": first_frame_conditioning_p,
        "train_text_encoder": train_text_encoder,
        "timestep_sampling_mode": timestep_sampling_mode
    }

def create_preset_selector_ui():
    """创建预设选择器UI组件"""
    with gr.Group():
        gr.Markdown("### 快速预设")
        with gr.Row():
            preset_selector = gr.Dropdown(
                label="训练预设", 
                choices=list(TRAINING_PRESETS.keys()),
                value=None,
                info="选择预定义的训练配置组合"
            )
            apply_preset_btn = gr.Button("应用预设")
            
    return {
        "preset_selector": preset_selector,
        "apply_preset_btn": apply_preset_btn
    }

def create_validation_params_ui():
    """创建验证参数UI组件"""
    with gr.Group():
        gr.Markdown("### 验证设置")
        with gr.Row():
            prompt = gr.Textbox(
                label="提示词", 
                placeholder="输入用于验证的提示词",
                info="用于生成验证视频的文本提示"
            )
            negative_prompt = gr.Textbox(
                label="负面提示词", 
                placeholder="输入负面提示词",
                value="worst quality",
                info="指定不希望出现的内容"
            )
            
        with gr.Row():
            inference_steps = gr.Slider(
                label="推理步数", 
                minimum=5, 
                maximum=50, 
                step=5,
                value=25,
                info="生成验证视频时的采样步数"
            )
            guidance_scale = gr.Slider(
                label="指导系数", 
                minimum=1.0, 
                maximum=15.0, 
                step=0.5,
                value=7.5,
                info="控制生成内容与提示词的相关性"
            )
            
    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale
    }

def collect_advanced_params(ui_components):
    """从UI组件收集高级参数"""
    config = DEFAULT_TRAINING_PARAMS.copy()
    
    # 安全获取组件值的函数
    def get_value(component_name):
        component = ui_components.get(component_name)
        # 确保我们获取的是值而不是组件对象
        if hasattr(component, "value"):
            return component.value
        return component
    
    # LoRA参数
    config["lora"]["rank"] = get_value("lora_rank")
    config["lora"]["alpha"] = get_value("lora_alpha")
    config["lora"]["dropout"] = get_value("lora_dropout")
    config["lora"]["target_modules"] = get_value("target_modules")
    
    # 优化参数
    config["optimization"]["learning_rate"] = get_value("learning_rate")
    config["optimization"]["scheduler_type"] = get_value("lr_scheduler")
    config["optimization"]["steps"] = get_value("steps")
    config["optimization"]["gradient_accumulation_steps"] = get_value("gradient_accumulation_steps")
    config["optimization"]["max_grad_norm"] = get_value("max_grad_norm")
    config["optimization"]["optimizer_type"] = get_value("optimizer_type")
    config["optimization"]["enable_gradient_checkpointing"] = get_value("enable_gradient_checkpointing")
    config["optimization"]["lr_warmup_steps"] = get_value("lr_warmup_steps")
    
    # 加速参数
    config["acceleration"]["mixed_precision_mode"] = get_value("mixed_precision_mode")
    config["acceleration"]["load_text_encoder_in_8bit"] = get_value("load_text_encoder_in_8bit")
    config["advanced_options"]["memory_efficient_attention"] = get_value("memory_efficient_attention")
    config["advanced_options"]["use_xformers"] = get_value("use_xformers")
    
    # 数据处理参数
    config["data"]["num_dataloader_workers"] = get_value("num_dataloader_workers")
    config["data"]["shuffle_batches"] = get_value("shuffle_batches")
    config["data"]["image_augmentation"]["enabled"] = get_value("enable_augmentation")
    config["data"]["image_augmentation"]["horizontal_flip_p"] = get_value("horizontal_flip_p")
    config["data"]["image_augmentation"]["vertical_flip_p"] = get_value("vertical_flip_p")
    config["data"]["image_augmentation"]["random_crop_p"] = get_value("random_crop_p")
    
    # 高级选项
    config["optimization"]["first_frame_conditioning_p"] = get_value("first_frame_conditioning_p")
    config["advanced_options"]["train_text_encoder"] = get_value("train_text_encoder")
    config["flow_matching"]["timestep_sampling_mode"] = get_value("timestep_sampling_mode")
    
    # 验证参数
    prompt = get_value("prompt")
    if prompt:
        config["validation"]["prompts"] = [prompt]
    config["validation"]["negative_prompt"] = get_value("negative_prompt")
    config["validation"]["inference_steps"] = get_value("inference_steps")
    config["validation"]["guidance_scale"] = get_value("guidance_scale")
    config["validation"]["interval"] = get_value("steps")
    
    # 检查点参数
    config["checkpoints"]["interval"] = get_value("steps")
    
    return config

def set_ui_values_from_config(ui_components, config):
    """根据配置设置UI组件的值"""
    # 直接创建更新列表，避免使用组件对象作为key
    updates = []
    
    # LoRA参数 - 显式指定保持可交互状态
    updates.append(gr.update(value=config["lora"]["rank"], interactive=True))
    updates.append(gr.update(value=config["lora"]["alpha"], interactive=True))
    updates.append(gr.update(value=config["lora"]["dropout"], interactive=True))
    updates.append(gr.update(value=config["lora"]["target_modules"], interactive=True))
    
    # 优化参数 - 显式指定保持可交互状态
    updates.append(gr.update(value=config["optimization"]["learning_rate"], interactive=True))
    updates.append(gr.update(value=config["optimization"]["scheduler_type"], interactive=True))
    updates.append(gr.update(value=config["optimization"]["steps"], interactive=True))
    updates.append(gr.update(value=config["optimization"]["gradient_accumulation_steps"], interactive=True))
    updates.append(gr.update(value=config["optimization"]["max_grad_norm"], interactive=True))
    updates.append(gr.update(value=config["optimization"]["optimizer_type"], interactive=True))
    updates.append(gr.update(value=config["optimization"]["enable_gradient_checkpointing"], interactive=True))
    updates.append(gr.update(value=config["optimization"]["lr_warmup_steps"], interactive=True))
    
    # 加速参数 - 显式指定保持可交互状态
    updates.append(gr.update(value=config["acceleration"]["mixed_precision_mode"], interactive=True))
    updates.append(gr.update(value=config["acceleration"]["load_text_encoder_in_8bit"], interactive=True))
    updates.append(gr.update(value=config["advanced_options"]["memory_efficient_attention"], interactive=True))
    updates.append(gr.update(value=config["advanced_options"]["use_xformers"], interactive=True))
    
    # 数据处理参数 - 显式指定保持可交互状态
    updates.append(gr.update(value=config["data"]["num_dataloader_workers"], interactive=True))
    updates.append(gr.update(value=config["data"]["shuffle_batches"], interactive=True))
    updates.append(gr.update(value=config["data"]["image_augmentation"]["enabled"], interactive=True))
    updates.append(gr.update(value=config["data"]["image_augmentation"]["horizontal_flip_p"], interactive=True))
    updates.append(gr.update(value=config["data"]["image_augmentation"]["vertical_flip_p"], interactive=True))
    updates.append(gr.update(value=config["data"]["image_augmentation"]["random_crop_p"], interactive=True))
    
    # 高级选项 - 显式指定保持可交互状态
    updates.append(gr.update(value=config["optimization"]["first_frame_conditioning_p"], interactive=True))
    updates.append(gr.update(value=config["advanced_options"]["train_text_encoder"], interactive=True))
    updates.append(gr.update(value=config["flow_matching"]["timestep_sampling_mode"], interactive=True))
    
    # 验证参数 - 显式指定保持可交互状态
    if config["validation"]["prompts"] and len(config["validation"]["prompts"]) > 0:
        updates.append(gr.update(value=config["validation"]["prompts"][0], interactive=True))
    else:
        updates.append(gr.update(interactive=True))
    updates.append(gr.update(value=config["validation"]["negative_prompt"], interactive=True))
    updates.append(gr.update(value=config["validation"]["inference_steps"], interactive=True))
    updates.append(gr.update(value=config["validation"]["guidance_scale"], interactive=True))
    
    # 预设选择器UI组件 (保持可交互状态)
    preset_name = config.get("_preset_name", "")
    updates.append(gr.update(value=preset_name, interactive=True))  # 预设选择器
    updates.append(gr.update(interactive=True))  # 预设应用按钮
    updates.append(gr.update(interactive=True))  # 其它按钮
    
    return updates
