import os
import sys
import torch
import psutil

# On Windows, PyTorch cu128 DLLs require System32 in the search path.
# This must be done before torch.cuda is accessed.
if sys.platform == "win32":
    os.add_dll_directory(r"C:\Windows\System32")


def _is_rocm() -> bool:
    """Returns True if PyTorch was built with ROCm (AMD GPU backend)."""
    return getattr(torch.version, "hip", None) is not None


def detect_hardware():
    """
    Detects the optimal hardware profile for the local node.
    Supports NVIDIA (CUDA/cu128) and AMD (ROCm/HIP) GPUs, with CPU fallback.
    Returns a configuration dict with device, batch_size, lora_rank,
    contribution_weight, and backend.
    """
    config = {}
    print(f"PyTorch Version : {torch.__version__}")

    # ROCm surfaces AMD GPUs through the same torch.cuda API
    gpu_available = torch.cuda.is_available()
    backend = "rocm" if _is_rocm() else "cuda"

    print(f"Backend         : {backend.upper()}")
    print(f"GPU Available   : {gpu_available}")

    if gpu_available:
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"GPU             : {props.name} ({vram_gb:.1f} GB VRAM)")

        # ROCm uses 'cuda' as the device string in PyTorch — same behaviour
        config['device'] = 'cuda'
        config['backend'] = backend

        if vram_gb >= 24:
            # High-end GPU (e.g. RTX 4090, RX 7900 XTX, A100)
            config['batch_size'] = 16
            config['lora_rank'] = 16
            config['contribution_weight'] = 2.0
        elif vram_gb >= 16:
            # Mid-range GPU (e.g. RTX 4080, RX 7900 XT, RTX 3090)
            config['batch_size'] = 4
            config['lora_rank'] = 8
            config['contribution_weight'] = 1.0
        elif vram_gb >= 7.5:
            # Consumer GPU (e.g. RTX 5060 Laptop 8 GB, RX 7700, RTX 3070)
            config['batch_size'] = 2
            config['lora_rank'] = 4
            config['contribution_weight'] = 0.75
        else:
            # Low VRAM GPU (e.g. Steam Deck iGPU, older mobile GPUs)
            config['batch_size'] = 1
            config['lora_rank'] = 2
            config['contribution_weight'] = 0.5
    else:
        # CPU fallback
        config['device'] = 'cpu'
        config['backend'] = 'cpu'
        sys_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        config['batch_size'] = 1
        config['lora_rank'] = 2
        config['contribution_weight'] = 0.1
        print(f"Warning: No GPU detected. Falling back to CPU with {sys_ram_gb:.1f} GB RAM. Training will be slow.")

    return config


if __name__ == "__main__":
    cfg = detect_hardware()
    print(f"\nHardware config : {cfg}")
