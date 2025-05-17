# LTX-Video-Trainer-GUI 安装详情记录

## 步骤1：获取代码仓库

首先需要从 GitHub 克隆仓库到本地：

```powershell
git clone https://github.com/chenpipi0807/LTX-Video-Trainer-GUI.git
cd LTX-Video-Trainer-GUI
```

执行成功后，你将在当前目录下看到项目文件夹。

## 步骤2：查看python版本
C:\ProgramData\anaconda3\python.exe --version（有anaconda3的运行这个，更推荐）
没有anaconda3的运行python --version 

我这里显示：
PS C:\LTX-Video-Trainer-GUI> python --version
Python 3.13.3

虽然项目推荐Python 3.12，但我们先尝试使用Python 3.13.3完成安装，如果遇到兼容性问题再考虑降级。

## 步骤3：检查CUDA环境

检查当前系统CUDA版本：

```powershell
nvidia-smi
```

输出结果：
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.94                 Driver Version: 560.94         CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060      WDDM  |   00000000:01:00.0  On |                  N/A |
|  0%   49C    P8             17W /  170W |     780MiB /  12288MiB |      5%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

**注意：** 当前系统CUDA版本为12.6，而项目推荐CUDA 12.8。我们将尝试安装CUDA 12.8兼容的PyTorch，但可能需要根据实际情况调整。

## 步骤4：安装PyTorch和CUDA支持

安装支持CUDA 12.8的PyTorch夜间版本：

```powershell
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**注意：**当尝试通过`requirements.txt`安装所有依赖时，遇到了PyTorch版本问题，我们将改为逻辑安装为主。

## 步骤5：安装必要依赖包

先安装PyYAML包（脚本运行需要）：

```powershell
pip install PyYAML
```

输出结果：
```
Collecting PyYAML
  Downloading PyYAML-6.0.2-cp313-cp313-win_amd64.whl.metadata (2.1 kB)
Downloading PyYAML-6.0.2-cp313-cp313-win_amd64.whl (156 kB)
Installing collected packages: PyYAML
Successfully installed PyYAML-6.0.2
```

接下来安装其他必要的包：

```powershell
pip install gradio rich typer pillow numpy opencv-python einops moviepy sentencepiece
```

安装结果：成功安装了所有UI相关和基础库。

## 步骤6：安装PyTorch和Diffusers

现在我们需要安装核心的PyTorch（支持CUDA）和Diffusers库，这些是项目运行的关键组件。

### 安装PyTorch夜间版

```powershell
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

安装结果：
```
Successfully installed mpmath-1.3.0 networkx-3.4.2 sympy-1.13.3 torch-2.8.0.dev20250515+cu128 torchaudio-2.6.0.dev20250516+cu128 torchvision-0.22.0.dev20250516+cu128
```

### 安装Diffusers最新版

```powershell
pip install git+https://github.com/huggingface/diffusers.git
```

安装结果：
```
Successfully installed diffusers-0.34.0.dev0 regex-2024.11.6 safetensors-0.5.3
```

## 步骤7：验证安装

运行`check_cuda.py`脚本验证CUDA环境：

```powershell
python check_cuda.py
```

输出结果：
```
PyTorch版本: 2.8.0.dev20250515+cu128
Torchvision版本: 0.22.0.dev20250516+cu128
CUDA可用: True
检测到GPU: NVIDIA GeForce RTX 3060
CUDA版本: 12.8
```

验证成功！现在我们已经安装了所有必要的依赖包，并确认PyTorch可以正确地使用CUDA 12.8。

## 步骤8：运行项目界面

下一步，我们使用脚本运行项目界面：

```powershell
python scripts\minimal_ui.py
```

运行结果：
- 界面成功启动在 http://127.0.0.1:7860
- 出现警告：`未找到diffusers模型: C:\LTX-Video-Trainer-GUI\models\LTX-Video-0.9.7-diffusers`

## 步骤9：下载所需模型

根据项目要求，我们需要下载以下模型文件：

### 使用下载脚本（推荐）

![微信截图_20250517184601](https://github.com/user-attachments/assets/26f48914-c604-4e56-bc7e-7577233597cb)


为了简化模型下载流程，我创建了一个下载脚本：

```powershell
python download_models.py
```

运行此脚本后，您可以选择：
1. 下载所有模型
2. 仅下载必需模型
3. 选择特定模型下载

脚本将自动处理下载和保存路径，大大简化安装流程。

### 手动下载方式

### 1. LTX-Video-0.9.7-diffusers (必需)

- 下载地址：[LTX-Video-0.9.7-diffusers](https://huggingface.co/a-r-r-o-w/LTX-Video-0.9.7-diffusers/tree/main)
- 保存位置：`.\models\LTX-Video-0.9.7-diffusers`
- 注意事项：必须保证文件夹结构完整，包含 text_encoder, tokenizer, transformer, vae 等子文件夹

### 2. LLaVA-NeXT-Video-7B-hf (用于视频标注，可选)

- 下载地址：[LLaVA-NeXT-Video-7B-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf/tree/main)
- 默认保存位置：`C:\Users\[用户名]\.cache\huggingface\hub\models--llava-hf--LLaVA-NeXT-Video-7B-hf\snapshots\[哈希值]`
- 注意：此路径因用户而异，使用脚本下载时会自动选择适当路径

### 3. T5-base (必需)

- 下载地址：[T5-base](https://huggingface.co/google-t5/t5-base/tree/main)
- 保存位置：`models\t5-base`



## 注意事项

目前测试的机型和环境有限，如果遇到问题请在github上提交issue，我会尽力帮助你解决。
当然本人时间精力有限，无法全力适配所有机型和环境，还请见谅。    


### 启动方式

**方式一：使用PowerShell脚本（推荐）**

- 如果使用标准Python环境：右键点击`run_system_python.ps1`，选择“使用PowerShell运行”
- 如果使用Anaconda环境：右键点击`run.ps1`，选择“使用PowerShell运行”

**方式二：手动运行命令**

- 标准Python环境：
  ```powershell
  python.exe scripts\minimal_ui.py
  ```

- Anaconda环境：
  ```powershell
  C:\ProgramData\anaconda3\python.exe scripts\minimal_ui.py
  ```
