# LTX-Video-Trainer 训练器使用指南

**感谢Lightricks团队开源**: [https://github.com/Lightricks/LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video-Trainer)

## 简介

LTX-Video-Trainer 是一款专业的LTX视频LoRA模型训练工具，支持通过简单的界面训练高质量的视频生成模型。本训练器提供了直观的图形界面，使用户能够轻松设置和启动训练流程，无需编写复杂代码。

![微信截图_20250515133328](https://github.com/user-attachments/assets/fa3c7338-86ea-4e40-b2b3-3bb5f34c3a20)

![微信截图_20250515133337](https://github.com/user-attachments/assets/8a68d60b-54a3-46c9-bc64-702ded0cb483)

![微信截图_20250515133357](https://github.com/user-attachments/assets/0ee8f8df-f65a-46d8-8deb-b8dfc406ad41)

![微信截图_20250515133424](https://github.com/user-attachments/assets/c69cbb66-02cc-4644-80a4-c3399e8db78e)

![微信截图_20250515133258](https://github.com/user-attachments/assets/287a039b-17bf-4c40-81d5-91007f2b6e17)

![image](https://github.com/user-attachments/assets/c20f502a-9f34-4e38-be77-88d2eb6b80a6)


## 说在前面

- 无论如何，让我们感谢开源精神
- 训练器做的很仓促，会有bug，欢迎提交issue或协助维护
- 环境相关问题，我已经尽可能适配24G及以下显存，但还是有风险
- 维护频率，工作原因，不会很高
- 安装问题，欢迎提交issue，我看到会回复

## 更新日志

- 05-18
- 优化了API推理返回错误的问题;


- 05-17
- 更新了安装指南，尽可能解决安装的老大难问题，请自行查阅安装文档吧~


- 05-16
- 新增了本地cuda环境的检查
- 新增了通过KIMI进行视频描述推理的进阶方案（现在支持手动选择字幕推理方式）
- 新增了视频未按照正确尺寸预处理导致的训练崩溃问题（目前的思路是按照短边缩放后中心裁切）

![image](https://github.com/user-attachments/assets/51cece61-c533-44d0-8de7-6d18c32f2ac4)



## 环境要求

- **操作系统**: Windows 10/11
- **Python**: Python 3.12 (Anaconda3)
- **GPU**: NVIDIA GPU (推荐至少 24GB 显存)
- **CUDA**: CUDA 12.8
- **硬盘空间**: 至少 50GB 可用空间

### 启动方式

**方式一：使用PowerShell脚本（推荐）**

- 如果使用标准Python环境：右键点击`run_system_python.ps1`，选择"使用PowerShell运行"
- 如果使用Anaconda环境：右键点击`run.ps1`，选择"使用PowerShell运行"

**方式二：手动运行命令**

- 标准Python环境：
  ```powershell
  python.exe scripts\minimal_ui.py
  ```

- Anaconda环境：
  ```powershell
  C:\ProgramData\anaconda3\python.exe scripts\minimal_ui.py
  ```


## 安装教程

详细的安装指南请参考: [安装指南](INSTALLATION_CN.md)

安装过程包括以下步骤:
1. 获取代码仓库
2. 安装PyTorch和Diffusers
3. 下载必要模型
4. 运行项目

## 必要模型下载

### 1. LTX-Video-0.9.7-diffusers (必需)

- **下载地址**: [LTX-Video-0.9.7-diffusers](https://huggingface.co/a-r-r-o-w/LTX-Video-0.9.7-diffusers/tree/main)
- **保存位置**: `models\LTX-Video-0.9.7-diffusers\`
- **注意事项**: 
  - 必须使用 diffusers 格式的模型
  - 文件结构不能错，文件不能缺少

### 2. LLaVA-NeXT-Video-7B-hf (用于视频标注)

- **下载地址**: [LLaVA-NeXT-Video-7B-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf/tree/main)
- **默认保存位置**: `C:\Users\[用户名]\.cache\huggingface\hub\models--llava-hf--LLaVA-NeXT-Video-7B-hf\snapshots\[哈希值]\`
- **说明**: 此模型用于视频自动标注，可以手动下载并放在上述目录

### 3. T5-base (必需)

- **下载地址**: [T5-base](https://huggingface.co/google-t5/t5-base/tree/main)
- **保存位置**: `models\t5-base\`
### 4. 模型结构

```
.\LTX-Video-Trainer-GUI\models
├─LTX-Video-0.9.7-diffusers
│  │  .gitattributes
│  │  model_index.json
│  │
│  ├─scheduler
│  │      scheduler_config.json
│  │
│  ├─text_encoder
│  │      config.json
│  │      model-00001-of-00004.safetensors
│  │      model-00002-of-00004.safetensors
│  │      model-00003-of-00004.safetensors
│  │      model-00004-of-00004.safetensors
│  │      model.safetensors.index.json
│  │
│  ├─tokenizer
│  │      added_tokens.json
│  │      special_tokens_map.json
│  │      spiece.model
│  │      tokenizer_config.json
│  │
│  ├─transformer
│  │      config.json
│  │      diffusion_pytorch_model-00001-of-00006.safetensors
│  │      diffusion_pytorch_model-00002-of-00006.safetensors
│  │      diffusion_pytorch_model-00003-of-00006.safetensors
│  │      diffusion_pytorch_model-00004-of-00006.safetensors
│  │      diffusion_pytorch_model-00005-of-00006.safetensors
│  │      diffusion_pytorch_model-00006-of-00006.safetensors
│  │      diffusion_pytorch_model.safetensors.index.json
│  │
│  └─vae
│          config.json
│          diffusion_pytorch_model.safetensors
│
└─t5-base
        .gitattributes.txt
        config.json
        flax_model.msgpack
        generation_config.json
        model.safetensors
        pytorch_model.bin
        README.md
        rust_model.ot
        spiece.model
        tf_model.h5
        tokenizer.json
        tokenizer_config.json
```

## 准备训练数据

1. 在 `train_date` 目录中创建以你的触发词命名的文件夹（例如 `APT`）
2. 将训练视频放入该文件夹中
3. 文件名应避免使用特殊符号
4. 默认触发词是 `APT`，可以修改不要搞中文或者奇怪符号

## 使用训练器

### 方法一: 一键训练流水线

1. 启动 UI
2. 在“一键训练流水线”标签下输入项目名称（与训练数据文件夹名称一致）
3. 选择合适的分辨率和帧数（根据您的显卡能力）
   - 支持多种分辨率选项，包括方形、横向和竖向分辨率
   - 帧数选项为24的倍数+1（如25、49、73等）
4. 选择预处理步骤：
   - 分场景：自动将长视频分割成独立场景
   - 标注视频：为每个场景生成文本描述
   - 预处理：生成训练所需的潜在表示和文本嵌入
5. 选择配置模板:
   - 对于基础训练，选择 `ltx_13b_lora_template`
   - 对于 8 位量化训练，选择 `ltx_13b_lora_int8-quanto_template`
6. 点击“开始一键训练”

### 方法二: 高级训练参数

1. 在"高级训练参数"标签中可以调整:
   - 学习率、批量大小、训练步数
   - LoRA 秩和丢弃率
   - 混合精度和量化方法
   - 验证设置和检查点间隔
2. 支持的量化选项:
   - `int8-quanto`: 8位量化，平衡速度和质量
   - `int4-quanto`: 4位量化，提高速度但可能影响质量
   - `int2-quanto`: 2位量化，最快但质量最低
3. 点击"保存参数到配置文件"后点击"开始高级训练"

### 方法三: 直接修改配置文件

对于更精细的控制，可以直接编辑配置文件:
```powershell
C:\ProgramData\anaconda3\python.exe scripts\train.py configs\ltx_13b_lora_int8-quanto.yaml
```

## 查看训练进度

训练开始后，进度会在命令行窗口实时显示。完成的模型将保存在 `outputs` 目录中。

## 新增功能和改进

### 1. 界面和用户体验改进

- **分辨率和帧数选择**：将分辨率和帧数分离成两个独立下拉列表
- **直观的分辨率标识**：分辨率选项添加[横版]、[竖版]、[方形]前缀，便于选择
- **新帧数选项**：所有帧数选项修改为24的倍数+1，支持更高帧数训练
- **训练步数扩展**：训练步数上限从2000增加到8000，支持更长时间训练

### 2. 预处理功能增强

- **可选预处理步骤**：用户可以自定义选择是否执行分场景、标注视频和预处理步骤
- **解决视频重复处理**：添加检查逻辑跳过已存在的场景目录和标题文件
- **标题文件优化**：改进JSON到TXT转换，仅使用文件名而非绝对路径
- **视频文件处理**：自动将视频文件从原始目录复制到标注目录
- **解决视频描述模型问题**：使用动态填充缺失类能出错的timm库问题

### 3. 训练流程稳定性提升

- **动态设备检测**：自动检测和选择可用GPU，优化处理速度
- **预处理参数格式修复**：确保分辨率参数传递为WxHxF格式
- **预处理数据路径智能处理**：支持多种预处理数据存放结构，包括`.precomputed`目录
- **错误处理改进**：强化了预处理和训练脚本的错误检测和日志输出

### 4. 离线应用支持

- **T5分词器离线加载**：在离线模式下使用本地T5分词器文件
- **虚拟VAE支持**：创建VAE兼容层，解决模型加载问题
- **编码兼容性**：解决Windows中文环境下的编码问题

## 常见问题与解决方案

1. **训练闪退**:
   - 检查显存是否不足，可尝试使用更激进的量化选项
   - 确认 CUDA 和 PyTorch 版本匹配

2. **找不到模型**:
   - 确认 LTX-Video-0.9.7-diffusers 已正确下载到指定目录
   - 检查模型文件结构是否完整

3. **VAE 错误**:
   - 确保使用最新版 diffusers 库
   - 检查是否使用了正确格式的 diffusers 模型

4. **UI 无法启动**:
   - 确认 Python 环境配置正确
   - 尝试使用提供的 `run.ps1` 脚本启动

## 高级功能

### 转换为 ComfyUI 格式

训练完成后，您可以将模型转换为 ComfyUI 格式:
1. 在"转换为ComfyUI格式"标签下
2. 输入训练好的模型路径（通常在 `outputs` 目录中）
3. 点击"转换为ComfyUI格式"

---

*更多信息和更新请关注官方文档和社区讨论。*
