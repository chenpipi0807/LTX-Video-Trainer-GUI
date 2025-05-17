# LTX-Video-Trainer-GUI 安装详情记录

-我得说这玩意会劝退很多人......但值得！
-以我一贯的作风，我尽可能写详细且易懂，安装相关的问题请自行查阅该文档，恕我无法一一回复。

## 步骤1：获取代码仓库

首先需要从 GitHub 克隆仓库到本地：

```powershell
git clone https://github.com/chenpipi0807/LTX-Video-Trainer-GUI.git
cd LTX-Video-Trainer-GUI
```

执行成功后，你将在当前目录下看到项目文件夹。

## 步骤2：安装正确的Python版本

**重要提示：** 项目需要使用Python 3.12.x版本，在Python 3.13.x上安装SentencePiece库会出现编译问题。

如果已安装Python 3.13.x，建议卸载并安装Python 3.12.3（最新稳定版）：

1. 卸载Python 3.13.x：通过控制面板或原安装程序卸载
2. 下载Python 3.12.3：从[Python官方下载页面](https://www.python.org/downloads/windows/)下载
3. 安装时勾选"Add Python 3.12 to PATH"选项

安装完成后验证版本：

```powershell
python --version
```

显示结果：
```
Python 3.12.3
```

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
  Downloading PyYAML-6.0.2-cp312-cp312-win_amd64.whl.metadata (2.1 kB)
Downloading PyYAML-6.0.2-cp312-cp312-win_amd64.whl (156 kB)
Installing collected packages: PyYAML
Successfully installed PyYAML-6.0.2
```

接下来安装其他必要的包：

```powershell
pip install gradio rich typer pillow numpy opencv-python einops moviepy
```

### 安装SentencePiece库（必需）

**特别注意：** SentencePiece库是指定项目使用T5模型的必要依赖，在Python 3.12上可以直接安装：

```powershell
pip install sentencepiece
```

安装结果：
```
Requirement already satisfied: sentencepiece in c:\users\pip\appdata\local\programs\python\python312\lib\site-packages (0.2.0)
```

确认安装成功后我们才能继续下一步。这是安装过程中最重要的一步，如果在Python 3.13版本上能会导致编译错误。

## 步骤6：简化安装方式

现在可以通过以下命令一次性安装所有依赖:

```powershell
pip install -r requirements.txt
```

对于特定需要从 GitHub 安装的库（如 diffusers），可以根据以下指导单独执行相应命令。


## 安装PyTorch和Diffusers

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

### 安装Transformers库（必需）

```powershell
pip install transformers
```

安装结果：
```
Successfully installed tokenizers-0.21.1 transformers-4.51.3
```

**注意：** Transformers库是预处理视频和图像的必要组件，缺少它会导致预处理脚本失败。

### 安装Loguru库（必需）

```powershell
pip install loguru
```

安装结果：
```
Successfully installed loguru-0.7.3 win32-setctime-1.2.0
```

**注意：** Loguru是项目使用的日志库，不安装此库会导致预处理脚本启动失败。

### 安装pillow-heif库（必需）

```powershell
pip install pillow-heif
```

安装结果：
```
Successfully installed pillow-heif-0.22.0
```

**注意：** pillow-heif库用于处理HEIF格式图像，预处理脚本依赖于这个库。

### 安装decord库（必需）

```powershell
pip install decord
```

安装结果：
```
Successfully installed decord-0.6.0
```

**注意：** decord是一个高效的视频处理库，在预处理和训练过程中用于视频解码。

### 安装optimum和optimum-quanto库（必需）

```powershell
pip install optimum optimum-quanto
```

安装结果：
```
Successfully installed optimum-1.25.3
Successfully installed ninja-1.11.1.4 optimum-quanto-0.2.7
```

**注意：** optimum和optimum-quanto库用于模型量化和优化，在训练过程中必不可少。

### 安装accelerate库（必需）

```powershell
pip install accelerate
```

安装结果：
```
Successfully installed accelerate-1.7.0 psutil-7.0.0
```

**注意：** accelerate库是用于分布式训练和混合精度训练的工具，训练脚本使用它来加速训练过程。

### 安装peft库（必需）

```powershell
pip install peft
```

安装结果：
```
Successfully installed peft-0.15.2
```

**注意：** peft库是用于参数高效微调（Parameter-Efficient Fine-Tuning）的库，它允许使用LoRA等技术微调大模型而只更新少量参数。

### 安装bitsandbytes库（必需）

```powershell
pip install -U bitsandbytes
```

安装结果：
```
Successfully installed bitsandbytes-0.45.5
```

**注意：** bitsandbytes库是用于模型量化的工具，视频训练过程中使用8位量化时必须安装该库。

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
- 默认保存位置：`C:\Users\[用户名]\.cache\huggingface\hub\models--llava-hf--LLaVA-NeXT-Video-7B-hf`
- 如果手动下载需要放置在`C:\Users\[用户名]\.cache\huggingface\hub\models--llava-hf--LLaVA-NeXT-Video-7B-hf\snapshots\[哈希值]`
- 注意：此路径因用户而异，使用脚本下载时会自动选择适当路径

### 3. T5-base (必需)

- 下载地址：[T5-base](https://huggingface.co/google-t5/t5-base/tree/main)
- 保存位置：`models\t5-base`


## 10: FFmpeg安装

FFmpeg是项目运行的必要组件，项目中使用了FFmpeg来处理视频文件。

### 自动安装FFmpeg

双击运行 install_ffmpeg_easy.bat，脚本会自动下载并安装FFmpeg。

### 手动安装FFmpeg

1. 下载FFmpeg：访问FFmpeg官方网站下载Windows版本的FFmpeg：https://ffmpeg.org/download.html
2. 解压FFmpeg：将下载的FFmpeg解压到项目目录的bin文件夹中。
3. 设置环境变量：运行set_ffmpeg_path.bat，将FFmpeg添加到系统环境变量中。  

  


## 11:启动GUI(真男人！恭喜你坚持到最后！)

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


### 说在最后

-写这个的时候是用的家里的3060，最后测试意外发现，我的老头乐显卡竟然能跑动的~

![bebaa70dd5037d5d266cba8661ac820](https://github.com/user-attachments/assets/b8e8b3c7-96fc-4425-ad20-de119ca13d86)




## 注意事项

目前测试的机型和环境有限，如果遇到问题请在github上提交issue，我会尽力帮助你解决。
当然本人时间精力有限，无法全力适配所有机型和环境，还请见谅。  

