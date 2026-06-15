import torch
import psutil

def detect_hardware():
    """
    Detects the optimal hardware profile for the local node.
    Returns a configuration dict with device, batch_size, and lora_rank.
    """
    config = {}

    if torch.cuda.is_available():
        config['device'] = 'cuda'
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if vram_gb >= 24:
            # High-end GPU
            config['batch_size'] = 16
            config['lora_rank'] = 16
            config['contribution_weight'] = 2.0
        elif vram_gb >= 16:
            # Mid-range GPU
            config['batch_size'] = 4
            config['lora_rank'] = 8
            config['contribution_weight'] = 1.0
        else:
            # Low-end GPU (e.g., Steam Deck or older GPU)
            config['batch_size'] = 2
            config['lora_rank'] = 4
            config['contribution_weight'] = 0.5
    else:
        # CPU Fallback
        config['device'] = 'cpu'
        sys_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # CPU nodes have severely limited throughput
        config['batch_size'] = 1
        config['lora_rank'] = 2
        config['contribution_weight'] = 0.1
        
        print(f"Warning: No GPU detected. Falling back to CPU with {sys_ram_gb:.1f}GB RAM. Training will be slow.")

    return config
