import torch
import psutil


def select_model_for_hardware():
    print("Evaluating local hardware to select optimal model...")
    
    if torch.cuda.is_available():
        # Get total VRAM in GB for the first GPU
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Detected GPU with {vram_gb:.1f} GB VRAM.")
        
        if vram_gb >= 16:
            # High-end hardware (MI300X, 4090, 3090, etc.)
            return {
                "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
                "reason": f"{vram_gb:.1f} GB VRAM is sufficient to load 8B parameters in 4-bit precision (~6.5GB) with plenty of room for LoRA training."
            }
        elif vram_gb >= 8:
            # Mid-range hardware (e.g., RTX 3070, RX 7600)
            return {
                "model_id": "microsoft/Phi-3-mini-4k-instruct",
                "reason": f"{vram_gb:.1f} GB VRAM can comfortably fit a 3.8B parameter model in 4-bit (~3.5GB)."
            }
        else:
            # Low-end hardware (Steam Deck, older GPUs)
            return {
                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "reason": f"{vram_gb:.1f} GB VRAM is highly constrained. Using TinyLlama 1.1B which needs ~1.2GB in 4-bit."
            }
    else:
        # CPU-only fallback
        sys_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"No GPU detected. Falling back to CPU. Available RAM: {sys_ram_gb:.1f} GB.")
        
        if sys_ram_gb >= 8:
            return {
                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "reason": "CPU training is extremely slow. Using TinyLlama 1.1B to prevent the system from completely locking up."
            }
        else:
            return {
                "model_id": "Qwen/Qwen1.5-0.5B",
                "reason": "Severely constrained CPU/RAM environment. Using ultra-small 0.5B model."
            }

if __name__ == "__main__":
    selection = select_model_for_hardware()
    print("\n--- Model Selection Result ---")
    print(f"Selected Model ID: {selection['model_id']}")
    print(f"Reason: {selection['reason']}")
