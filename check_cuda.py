import torch
import torchvision

print(f"PyTorch版本: {torch.__version__}")
print(f"Torchvision版本: {torchvision.__version__}")
print(f"CUDA可用性: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"检测到的GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
else:
    print("警告: CUDA不可用，将使用CPU")
