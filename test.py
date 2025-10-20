import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA is available: {torch.cuda.is_available()}")

try:
    # 这个命令会强制CUDA初始化，并揭示真正的错误
    print("\n正在尝试强制初始化 CUDA...")
    torch.tensor([1.0]).cuda(0)
    print("CUDA 初始化成功!")
    print(f"PyTorch 编译时使用的 CUDA 版本: {torch.version.cuda}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")

except Exception as e:
    print("\n!!! CUDA 初始化失败，具体错误信息如下:")
    print(e)