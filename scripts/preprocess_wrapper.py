#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预处理脚本的包装器，解决Windows中文环境下的编码问题
"""

import os
import sys
import subprocess

# 强制设置编码为UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"

if __name__ == "__main__":
    # 从命令行获取参数，然后移除脚本名称
    args = sys.argv[1:]
    
    # 构建调用原始预处理脚本的命令
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocess_script = os.path.join(script_dir, "preprocess_dataset.py")
    
    cmd = [sys.executable, preprocess_script] + args
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    # 输出即将执行的命令
    print(f"执行预处理命令: {' '.join(cmd)}")
    
    # 启动子进程，继承标准输入输出
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        # 重要：不要设置encoding，让子进程直接继承标准输入输出
        # 解决Windows中文环境下的编码问题
        bufsize=1,
        universal_newlines=True
    )
    
    # 等待进程完成并获取返回码
    return_code = process.wait()
    
    # 退出时使用相同的返回码
    sys.exit(return_code)
