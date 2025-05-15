#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预处理修复脚本 - 强制使用单线程处理
解决Can't pickle local object错误
同时确保分辨率格式为WxHxF
"""

import os
import sys
import subprocess
import traceback

def main():
    try:
        # 获取命令行参数
        args = sys.argv[1:]
        
        if len(args) == 0:
            print("错误: 缺少参数。请提供数据集路径和其他必要参数。")
            return 1
        
        # 输出原始命令行参数，方便调试
        print(f"原始参数: {' '.join(args)}")
        
        # 查找是否有帧数参数
        frames_value = None
        frames_index = -1
        for i, arg in enumerate(args):
            if arg == "--frames" and i + 1 < len(args):
                frames_value = args[i + 1]
                frames_index = i
                print(f"发现帧数参数: --frames {frames_value}")
                break
        
        # 处理参数，固定num_workers=0
        modified_args = []
        i = 0
        while i < len(args):
            if args[i] == "--resolution-buckets" and i + 1 < len(args):
                # 检查分辨率格式
                resolution = args[i + 1]
                x_count = resolution.count("x")
                
                if x_count == 2:  # 已经是WxHxF格式
                    print(f"检测到已经是WxHxF格式: {resolution}")
                    # 直接使用原始分辨率
                    modified_args.append(args[i])  # --resolution-buckets
                    modified_args.append(resolution)
                elif x_count == 1:  # WxH格式
                    # 只保留WxH格式，不再主动添加帧数
                    # 因为preprocess_dataset.py现在支持单独的--frames参数
                    print(f"检测到WxH格式分辨率: {resolution}") 
                    if frames_value:
                        print(f"将使用单独的帧数参数: --frames {frames_value}")
                    else:
                        print("没有发现帧数参数，将使用预处理脚本的默认值")
                    
                    # 直接使用原始分辨率
                    modified_args.append(args[i])  # --resolution-buckets
                    modified_args.append(resolution)
                else:
                    # 其他情况，直接使用原始参数
                    print(f"未识别的分辨率格式: {resolution}，直接使用")
                    modified_args.append(args[i])  # --resolution-buckets
                    modified_args.append(resolution)
                
                i += 2
            elif args[i] == "--num-workers" and i + 1 < len(args):
                # 发现num_workers参数，强制设置为0
                modified_args.append(args[i])  # --num-workers
                modified_args.append("0")  # 强制为0
                print(f"修复多进程问题: 将--num-workers参数值从 {args[i+1]} 改为 0")
                i += 2
            else:
                # 其他参数保持不变（包括--frames参数）
                modified_args.append(args[i])
                
                # 检查是否有值跟随当前参数
                if i + 1 < len(args) and not args[i+1].startswith("--"):
                    modified_args.append(args[i+1])
                    i += 2
                else:
                    # 没有值跟随，单独的标志
                    i += 1
        
        # 如果没有找到num_workers参数，添加它
        if "--num-workers" not in modified_args:
            modified_args.append("--num-workers")
            modified_args.append("0")
            print("添加参数: --num-workers 0")
        
        # 获取原始预处理脚本路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        preprocess_script = os.path.join(script_dir, "preprocess_dataset.py")
        
        # 检查脚本是否存在
        if not os.path.exists(preprocess_script):
            print(f"错误: 找不到原始预处理脚本: {preprocess_script}")
            return 1
        
        # 构建新命令
        cmd = [sys.executable, preprocess_script] + modified_args
        
        # 打印将要执行的命令
        print(f"执行预处理命令: {' '.join(cmd)}")
        
        # 设置环境变量
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        try:
            # 执行命令
            process = subprocess.Popen(
                cmd, 
                stdout=sys.stdout, 
                stderr=sys.stderr,
                env=env,
                universal_newlines=True
            )
            
            # 等待进程完成并获取返回码
            return_code = process.wait()
            
            # 输出结果
            if return_code != 0:
                print(f"预处理脚本执行失败，返回码：{return_code}")
            else:
                print("预处理脚本成功执行完成！")
                
            return return_code
            
        except Exception as e:
            print(f"执行预处理脚本时出错: {str(e)}")
            traceback.print_exc()
            return 1
            
    except Exception as e:
        print(f"脚本内部错误: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
