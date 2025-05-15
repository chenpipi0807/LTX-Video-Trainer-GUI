import subprocess
import os
import sys
from pathlib import Path

# 确保当前目录是项目根目录
os.chdir(Path(__file__).parent.parent)

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# 定义轻量级训练配置
CONFIG = """
model:
  model_source: LTXV_13B_097_DEV
  training_mode: lora
  load_checkpoint: null

lora:
  rank: 64  # 降低了LoRA秩，减少内存使用
  alpha: 64
  dropout: 0.0
  target_modules:
    - to_k
    - to_q
    - to_v
    - to_out.0

optimization:
  learning_rate: 0.0002
  steps: 300  # 减少训练步数
  batch_size: 1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  optimizer_type: adamw
  scheduler_type: linear
  scheduler_params: {}
  enable_gradient_checkpointing: true
  first_frame_conditioning_p: 0.5

acceleration:
  mixed_precision_mode: bf16
  quantization: null
  load_text_encoder_in_8bit: false
  compile_with_inductor: false  # 禁用编译以避免潜在的问题
  compilation_mode: reduce-overhead

data:
  preprocessed_data_root: APT_scenes/.precomputed
  num_dataloader_workers: 0  # 保持0以避免序列化问题

validation:
  prompts:
    - "在视频中，一个女性角色正在跳舞"
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"
  images: null
  video_dims: [512, 512, 41]  # 使用更小的尺寸和更短的视频长度
  seed: 42
  inference_steps: 25  # 减少推理步数以加快验证
  interval: 100  # 更频繁地创建验证视频
  videos_per_prompt: 1
  guidance_scale: 3.5

checkpoints:
  interval: 100  # 更频繁地保存检查点
  keep_last_n: -1

flow_matching:
  timestep_sampling_mode: shifted_logit_normal
  timestep_sampling_params: {}

seed: 42
output_dir: outputs/train_lite_output
"""

def main():
    # 创建配置目录
    os.makedirs("configs", exist_ok=True)
    
    # 写入轻量级配置
    config_path = "configs/train_lite_config.yaml"
    with open(config_path, "w") as f:
        f.write(CONFIG)
    
    print(f"创建了轻量级训练配置: {config_path}")
    
    # 创建输出目录
    output_dir = "outputs/train_lite_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保预处理数据目录存在
    if not os.path.exists("APT_scenes/.precomputed"):
        print("错误: 预处理数据目录不存在。请先运行预处理步骤。")
        return
    
    print("开始轻量级训练...")
    print("这将使用较少的步数和更小的LoRA参数进行训练")
    print("输出将保存到:", output_dir)
    
    # 运行训练命令
    try:
        # 使用正确的训练脚本
        command = [
            sys.executable, 
            "scripts/train.py", 
            config_path
            # 移除不支持的verbose选项
        ]
        
        print("执行命令:", " ".join(command))
        
        # 使用实时输出运行命令
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 实时打印输出
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print("训练完成!")
            print(f"检查点已保存到: {output_dir}/checkpoints")
            print(f"示例视频已保存到: {output_dir}/samples")
        else:
            print(f"训练失败，返回代码: {process.returncode}")
            
    except Exception as e:
        print(f"执行训练时出错: {e}")

if __name__ == "__main__":
    main()
