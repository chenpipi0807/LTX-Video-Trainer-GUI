import subprocess
import time
import os
import psutil
import datetime

def get_gpu_info():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], shell=True)
        lines = result.decode('utf-8').strip().split('\n')
        gpu_info = []
        for i, line in enumerate(lines):
            values = [float(val.strip()) for val in line.split(',')]
            gpu_info.append({
                'id': i,
                'utilization': values[0],
                'memory_used': values[1],
                'memory_total': values[2],
                'temperature': values[3]
            })
        return gpu_info
    except:
        return None

def get_process_info(process_name="python"):
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            if process_name.lower() in proc.info['name'].lower():
                cpu = proc.info['cpu_percent']
                mem = proc.info['memory_percent']
                command = " ".join(proc.cmdline())
                if "run_pipeline" in command or "minimal_ui" in command or "easy_trainer" in command:
                    processes.append({
                        'pid': proc.info['pid'],
                        'cpu': cpu,
                        'memory': mem,
                        'command': command
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("LTX-Video-Trainer 监控器")
    print("按Ctrl+C退出")
    
    try:
        while True:
            clear_screen()
            print(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("\n===== GPU状态 =====")
            gpu_info = get_gpu_info()
            
            if not gpu_info:
                print("无法获取GPU信息。请确保安装了NVIDIA驱动。")
            else:
                for gpu in gpu_info:
                    print(f"GPU {gpu['id']}:")
                    print(f"  使用率: {gpu['utilization']:.1f}%")
                    print(f"  内存: {gpu['memory_used']:.0f}MB / {gpu['memory_total']:.0f}MB ({gpu['memory_used']/gpu['memory_total']*100:.1f}%)")
                    print(f"  温度: {gpu['temperature']:.0f}°C")
            
            print("\n===== 训练进程 =====")
            processes = get_process_info()
            if not processes:
                print("未检测到LTX训练进程")
            else:
                for proc in processes:
                    print(f"PID: {proc['pid']}")
                    print(f"CPU: {proc['cpu']:.1f}%")
                    print(f"内存: {proc['memory']:.1f}%")
                    print(f"命令: {proc['command'][:80]}..." if len(proc['command']) > 80 else f"命令: {proc['command']}")
                    print()
            
            print("\n如果GPU使用率和内存使用量高，说明训练正在进行中。")
            print("如果系统占用很高但没有进展，模型可能是在加载阶段（特别是13B参数模型）。")
            print("模型加载可能需要5-10分钟，请耐心等待。")
            print("\n每5秒刷新一次...\n")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n监控器已关闭。")

if __name__ == "__main__":
    main()
