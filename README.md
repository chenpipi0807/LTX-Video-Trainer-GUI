# LTX-Video-Trainer 训练器使用指南

## 简介

LTX-Video-Trainer 是为LTX视频lora模型训练工具，支持通过简单的界面训练 LoRA 模型用于视频生成。本训练器提供了直观的 GUI 界面，使用户能够轻松设置和启动训练流程，无需编写复杂代码。

![微信截图_20250515133328](https://github.com/user-attachments/assets/fa3c7338-86ea-4e40-b2b3-3bb5f34c3a20)

![微信截图_20250515133337](https://github.com/user-attachments/assets/8a68d60b-54a3-46c9-bc64-702ded0cb483)

![微信截图_20250515133357](https://github.com/user-attachments/assets/0ee8f8df-f65a-46d8-8deb-b8dfc406ad41)

![微信截图_20250515133424](https://github.com/user-attachments/assets/c69cbb66-02cc-4644-80a4-c3399e8db78e)

![微信截图_20250515133258](https://github.com/user-attachments/assets/287a039b-17bf-4c40-81d5-91007f2b6e17)






## 环境要求

- **操作系统**: Windows 10/11
- **Python**: Python 3.12 (Anaconda3)
- **GPU**: NVIDIA GPU (推荐至少 24GB 显存)
- **CUDA**: CUDA 12.8
- **硬盘空间**: 至少 50GB 可用空间

## 快速启动

1. **简单启动方式**:
   - 右键点击 `run.ps1` 文件
   - 选择"使用 PowerShell 运行"

2. **手动启动命令**:
   ```powershell
   C:\ProgramData\anaconda3\python.exe scripts\minimal_ui.py
   ```

## 环境安装

### 安装 PyTorch 与 CUDA 支持

必须安装适配 CUDA 12.8 的 PyTorch 夜间版本:
```powershell
C:\ProgramData\anaconda3\python.exe -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 安装最新版 Diffusers 库

必须安装 GitHub 上的最新版本，官方发布版本无法正常运行 VAE:
```powershell
C:\ProgramData\anaconda3\python.exe -m pip install git+https://github.com/huggingface/diffusers.git
```

### 验证安装

```powershell
C:\ProgramData\anaconda3\python.exe check_cuda.py
```

正确输出应类似:
```
PyTorch版本: 2.8.0.dev20250511+cu128
Torchvision版本: 0.22.0.dev20250512+cu128
CUDA可用性: True
检测到的GPU: NVIDIA GeForce RTX 4090
CUDA版本: 12.8
```

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

### 3. T5-base (可选)

- **下载地址**: [T5-base](https://huggingface.co/google-t5/t5-base/tree/main)
- **保存位置**: `models\t5-base\`

## 准备训练数据

1. 在 `train_date` 目录中创建以你的触发词命名的文件夹（例如 `APT`）
2. 将训练视频放入该文件夹中
3. 文件名应避免使用特殊符号
4. 默认触发词是 `APT`，建议保持此名称以免出错

## 使用训练器

### 方法一: 一键训练

1. 启动 UI
2. 在"一键流水线"标签下输入项目名称（与训练数据文件夹名称一致）
3. 选择合适的分辨率（根据您的显卡能力）
4. 选择配置模板:
   - 对于基础训练，选择 `ltx_13b_lora_template`
   - 对于 8 位量化训练，选择 `ltx_13b_lora_int8-quanto_template`
5. 点击"开始一键训练"

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

## 常见问题

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
