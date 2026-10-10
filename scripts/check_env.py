"""Sanity check: PyTorch sees the GPU and can run a kernel on it."""
import platform

import torch

print(f"python      {platform.python_version()}")
print(f"torch       {torch.__version__} (cuda {torch.version.cuda})")
print(f"cuda avail  {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available - check the Windows NVIDIA driver and the torch wheel.")

dev = torch.device("cuda")
print(f"gpu         {torch.cuda.get_device_name(dev)}  sm_{''.join(map(str, torch.cuda.get_device_capability(dev)))}")
x = torch.randn(2048, 2048, device=dev)
y = (x @ x).sum().item()  # fails here if the wheel has no kernels for this GPU
print(f"matmul ok   {y:.1f}")

try:
    import coremltools as ct
    print(f"coremltools {ct.__version__}")
except ImportError:
    print("coremltools not installed")
