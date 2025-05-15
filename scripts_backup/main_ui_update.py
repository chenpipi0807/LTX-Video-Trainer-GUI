def main():
    """创建Gradio UI界面"""
    # 读取配置文件列表
    config_files = get_config_files()
    default_config_path = os.path.join(CONFIG_DIR, "default_config.yaml")
    
    # 检测GPU信息
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_info = f"**GPU: {gpu_name} | 显存: {gpu_memory:.1f} GB**"
    else:
        gpu_info = "**⚠️ 未检测到GPU。训练需要NVIDIA GPU支持。**"
    
    with gr.Blocks(title="LTX-Video-Trainer 专业版") as app:
        gr.Markdown("# 🚀 LTX-Video-Trainer 专业版")
        gr.Markdown("### 提供完整的视频模型训练和转换功能")
        
        # 显示GPU和模型状态
        gr.Markdown(gpu_info)
        gr.Markdown("## 模型状态")
        model_status = check_models()
        gr.Markdown("\n".join(model_status))
        
        with gr.Tabs():
            # 离线训练模式标签页
            with gr.TabItem("本地离线训练"):
                gr.Markdown("### 🔥 完全离线模式 - 不需要下载任何资源")
                gr.Markdown("该模式使用本地模型文件直接生成LoRA权重，简化训练流程并避免任何网络请求")
                
                with gr.Row():
                    with gr.Column():
                        offline_basename = gr.Textbox(
                            label="项目名称", 
                            placeholder="输入项目名称，如APT，数据集应放在train_date/APT目录中"
                        )
                        gr.Markdown("支持的数据集位置：**train_date/{项目名}** 或 **{项目名}_raw**")
                        offline_model_size = gr.Radio(
                            label="模型大小", 
                            choices=["2B", "13B"], 
                            value="2B",
                            info="2B模型需要较少的显存，13B模型需要更多显存但质量更高"
                        )
                        offline_resolution = gr.Dropdown(
                            label="分辨率", 
                            choices=RESOLUTIONS,
                            value="768x768x25",
                            info="格式为宽x高x帧数，帧数较少训练更快"
                        )
                        offline_rank = gr.Slider(
                            label="LoRA秩 (Rank)", 
                            minimum=1, 
                            maximum=128, 
                            value=32,
                            info="值越大，适应性越强但需要更多显存"
                        )
                        offline_steps = gr.Slider(
                            label="训练步数", 
                            minimum=5, 
                            maximum=200, 
                            value=50,
                            info="步数越多，训练时间越长，但效果可能更好"
                        )
                        
                        # 添加保存配置文件选项
                        offline_save_config = gr.Checkbox(
                            label="保存为配置文件",
                            value=False,
                            info="勾选可将当前参数保存为配置文件供以后使用"
                        )
                        offline_config_name = gr.Textbox(
                            label="配置名称",
                            placeholder="自定义配置名称",
                            visible=False
                        )
                        
                        offline_button = gr.Button(
                            "开始本地离线训练", 
                            variant="primary"
                        )
                        
                        # 控制配置名称框的显示/隐藏
                        offline_save_config.change(
                            lambda x: gr.update(visible=x),
                            inputs=[offline_save_config],
                            outputs=[offline_config_name]
                        )
                    
                    with gr.Column():
                        offline_status = gr.Textbox(
                            label="训练日志", 
                            lines=25,
                            max_lines=30,
                            autoscroll=True
                        )
                
                # 自定义点击处理函数，支持保存配置文件
                def offline_train_with_config(basename, model_size, resolution, rank, steps, save_config, config_name, status):
                    # 如果勾选了保存配置文件
                    if save_config and config_name:
                        # 创建配置内容
                        resolution_parts = resolution.split('x')
                        video_dims = [int(x) for x in resolution_parts] if len(resolution_parts) == 3 else [768, 768, 25]
                        
                        config = {
                            "model": {
                                "training_mode": "lora",
                                "load_checkpoint": None
                            },
                            "lora": {
                                "rank": rank,
                                "alpha": rank,
                                "dropout": 0.0,
                                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"]
                            },
                            "optimization": {
                                "learning_rate": 0.0002,
                                "steps": steps,
                                "batch_size": 1,
                                "gradient_accumulation_steps": 1,
                                "max_grad_norm": 1.0,
                                "optimizer_type": "adamw",
                                "scheduler_type": "linear",
                                "scheduler_params": {},
                                "enable_gradient_checkpointing": True,
                                "first_frame_conditioning_p": 0.5
                            },
                            "acceleration": {
                                "mixed_precision_mode": "fp16",
                                "quantization": None,
                                "load_text_encoder_in_8bit": False,
                                "compile_with_inductor": False,
                                "compilation_mode": "default"
                            },
                            "validation": {
                                "prompts": [basename],
                                "negative_prompt": "worst quality",
                                "images": None,
                                "video_dims": video_dims,
                                "seed": 42,
                                "inference_steps": 5,
                                "interval": steps,
                                "videos_per_prompt": 1,
                                "guidance_scale": 3.5
                            },
                            "checkpoints": {
                                "interval": steps,
                                "keep_last_n": 1
                            },
                            "flow_matching": {
                                "timestep_sampling_mode": "shifted_logit_normal",
                                "timestep_sampling_params": {}
                            },
                            "seed": 42
                        }
                        
                        # 保存配置文件
                        config_path = os.path.join(CONFIG_DIR, f"{config_name}.yaml")
                        os.makedirs(CONFIG_DIR, exist_ok=True)
                        save_config(config, config_path)
                        logger.info(f"配置文件已保存: {config_path}")
                        
                        # 更新状态
                        if hasattr(status, 'update'):
                            current_status = status.value if hasattr(status, 'value') else ""
                            status.update(value=f"配置已保存到: {config_path}\n\n{current_status}")
                    
                    # 运行训练
                    return run_offline_training(basename, model_size, resolution, rank, steps, status)
                
                offline_button.click(
                    fn=offline_train_with_config,
                    inputs=[
                        offline_basename, 
                        offline_model_size, 
                        offline_resolution, 
                        offline_rank, 
                        offline_steps,
                        offline_save_config,
                        offline_config_name,
                        offline_status
                    ],
                    outputs=offline_status
                )
            
            # 数据预处理标签页
            with gr.TabItem("数据预处理"):
                gr.Markdown("### 📊 视频数据预处理工具")
                gr.Markdown("该工具用于处理原始视频数据，生成训练所需的场景和标题")
                
                with gr.Row():
                    with gr.Column():
                        preprocess_dataset = gr.Textbox(
                            label="项目名称", 
                            placeholder="输入项目名称，如APT，数据集应放在train_date/APT目录中"
                        )
                        preprocess_resolution = gr.Dropdown(
                            label="分辨率", 
                            choices=RESOLUTIONS,
                            value="768x768x25",
                            info="格式为宽x高x帧数"
                        )
                        preprocess_id_token = gr.Textbox(
                            label="ID标记 (LoRA触发词)", 
                            placeholder="例如: <特效>，留空则不使用特殊触发词",
                            value=""
                        )
                        preprocess_decode = gr.Checkbox(
                            label="解码视频进行验证", 
                            value=True,
                            info="开启可验证视频帧解码是否正确，但会减慢处理速度"
                        )
                        preprocess_button = gr.Button(
                            "开始预处理", 
                            variant="primary"
                        )
                    
                    with gr.Column():
                        preprocess_status = gr.Textbox(
                            label="预处理日志", 
                            lines=25,
                            max_lines=30,
                            autoscroll=True
                        )
                
                preprocess_button.click(
                    fn=run_preprocessing,
                    inputs=[
                        preprocess_dataset,
                        preprocess_resolution,
                        preprocess_id_token,
                        preprocess_decode,
                        preprocess_status
                    ],
                    outputs=preprocess_status
                )
            
            # 转换为ComfyUI格式标签页
            with gr.TabItem("转换为ComfyUI格式"):
                gr.Markdown("### 🔄 模型格式转换工具")
                gr.Markdown("将训练好的模型转换为ComfyUI兼容格式，便于在ComfyUI中使用")
                
                with gr.Row():
                    with gr.Column():
                        convert_input = gr.Textbox(
                            label="输入模型路径", 
                            placeholder="训练好的模型权重路径，例如outputs/APT_offline_training/lora_weights/adapter_model.safetensors"
                        )
                        convert_output = gr.Textbox(
                            label="输出路径 (可选)", 
                            placeholder="留空则自动生成输出路径",
                            value=""
                        )
                        convert_button = gr.Button(
                            "转换为ComfyUI格式", 
                            variant="primary"
                        )
                    
                    with gr.Column():
                        convert_status = gr.Textbox(
                            label="转换日志", 
                            lines=25,
                            max_lines=30,
                            autoscroll=True
                        )
                
                convert_button.click(
                    fn=convert_to_comfyui,
                    inputs=[
                        convert_input,
                        convert_output,
                        convert_status
                    ],
                    outputs=convert_status
                )
            
            # 配置管理标签页
            with gr.TabItem("配置管理"):
                gr.Markdown("### ⚙️ 配置文件管理")
                gr.Markdown("查看和编辑训练参数配置文件")
                
                with gr.Row():
                    with gr.Column():
                        config_list = gr.Dropdown(
                            label="选择配置文件",
                            choices=list(config_files.keys()),
                            info="选择一个已存在的配置文件进行查看或编辑"
                        )
                        config_refresh = gr.Button("刷新列表")
                        
                        def refresh_configs():
                            configs = get_config_files()
                            return gr.update(choices=list(configs.keys()))
                        
                        config_refresh.click(
                            fn=refresh_configs,
                            inputs=[],
                            outputs=[config_list]
                        )
                    
                    with gr.Column():
                        config_content = gr.TextArea(
                            label="配置内容",
                            lines=20,
                            info="配置文件的YAML格式内容"
                        )
                
                def load_config_content(config_name):
                    if not config_name:
                        return ""
                    config_files = get_config_files()
                    if config_name in config_files:
                        path = config_files[config_name]
                        try:
                            with open(path, 'r', encoding='utf-8') as f:
                                return f.read()
                        except Exception as e:
                            return f"读取配置文件失败: {str(e)}"
                    return "未找到配置文件"
                
                config_list.change(
                    fn=load_config_content,
                    inputs=[config_list],
                    outputs=[config_content]
                )
                
                with gr.Row():
                    config_save = gr.Button("保存修改")
                    config_delete = gr.Button("删除配置", variant="stop")
                
                def save_config_changes(config_name, content):
                    if not config_name:
                        return "请先选择一个配置文件"
                    
                    config_files = get_config_files()
                    if config_name in config_files:
                        path = config_files[config_name]
                        try:
                            with open(path, 'w', encoding='utf-8') as f:
                                f.write(content)
                            return f"配置 {config_name} 已保存"
                        except Exception as e:
                            return f"保存配置文件失败: {str(e)}"
                    return "未找到配置文件"
                
                def delete_config_file(config_name):
                    if not config_name:
                        return "请先选择一个配置文件", gr.update(choices=list(get_config_files().keys()))
                    
                    config_files = get_config_files()
                    if config_name in config_files:
                        path = config_files[config_name]
                        try:
                            os.remove(path)
                            new_configs = get_config_files()
                            return f"配置 {config_name} 已删除", gr.update(choices=list(new_configs.keys()))
                        except Exception as e:
                            return f"删除配置文件失败: {str(e)}", gr.update(choices=list(config_files.keys()))
                    return "未找到配置文件", gr.update(choices=list(config_files.keys()))
                
                config_result = gr.Textbox(label="操作结果")
                
                config_save.click(
                    fn=save_config_changes,
                    inputs=[config_list, config_content],
                    outputs=[config_result]
                )
                
                config_delete.click(
                    fn=delete_config_file,
                    inputs=[config_list],
                    outputs=[config_result, config_list]
                )
        
        # 页脚信息
        gr.Markdown("---")
        gr.Markdown("### 📝 使用说明")
        gr.Markdown("""
        1. **数据集准备**:
           - 数据集应放在 `train_date/{项目名}` 或 `{项目名}_raw` 目录中
           - 可以通过"数据预处理"标签页处理原始数据
        
        2. **训练流程**:
           - 选择模型大小、分辨率和训练参数
           - 点击"开始本地离线训练"按钮
           - 训练完成后，LoRA文件将保存在 `outputs/{项目名}_offline_training/lora_weights` 目录
        
        3. **使用配置文件**:
           - 可以在"配置管理"标签页查看和编辑配置文件
           - 训练时可选择保存当前参数为配置文件供以后使用
        
        4. **ComfyUI整合**:
           - 训练完成后，可通过"转换为ComfyUI格式"标签页转换模型格式
           - 转换后的模型可直接用于ComfyUI的视频工作流
        """)
        
    # 启动UI
    app.launch(server_name="127.0.0.1", server_port=7860)
