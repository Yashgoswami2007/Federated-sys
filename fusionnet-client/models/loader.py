import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import psutil

# Federation model constant — must match config.yaml across ALL nodes.
# TinyLlama-1.1B shares the Llama transformer architecture so AFLoRA injection
# and FedAvg aggregation work without any code changes.
FEDERATION_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class DeviceDetector:
    @staticmethod
    def detect_hardware():
        """
        Detects the local hardware profile.

        Returns:
            (profile_name, device_type) tuple, e.g. ("CPU_only", "cpu")

        Profile names match the keys in config.yaml device_profiles, which
        control LoRA rank, batch size, and contribution weight.
        The model itself is ALWAYS FEDERATION_MODEL_ID regardless of profile.
        """
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

            if "mi300x" in gpu_name or vram_gb >= 100:
                return "MI300X", "cuda"          # ROCm surfaces as cuda in PyTorch
            elif "7900" in gpu_name or vram_gb >= 20:
                return "RX_7900_XTX", "cuda"
            elif vram_gb >= 7.5:
                return "Steam_Deck", "cuda"      # Consumer GPU (8 GB class)
            else:
                return "Steam_Deck", "cuda"      # Fallback for smaller GPUs

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Steam_Deck", "mps"

        else:
            sys_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            print(f"No GPU detected. CPU-only mode. System RAM: {sys_ram_gb:.1f} GB")
            return "CPU_only", "cpu"


def load_model(model_name: str = None, quantization_type: str = "nf4"):
    """
    Loads the federation model with hardware-appropriate precision.

    GPU nodes  : 4-bit NF4 (bitsandbytes) — TinyLlama uses ~1.2 GB VRAM
    CPU nodes  : FP32, no quantization    — TinyLlama uses ~2.5 GB RAM

    Args:
        model_name (str)       : HuggingFace model ID. If None or "auto",
                                 defaults to FEDERATION_MODEL_ID (TinyLlama).
        quantization_type (str): bitsandbytes quant type for GPU path (default: "nf4").

    Returns:
        model         : Frozen base model ready for AFLoRA injection.
        tokenizer     : Associated tokenizer.
        device_profile: Hardware profile string (e.g. "CPU_only", "MI300X").
    """
    # Always resolve to the federation model — per-device model choice
    # breaks FedAvg aggregation (A matrix shape mismatch).
    if model_name is None or model_name == "auto":
        model_name = FEDERATION_MODEL_ID

    if model_name != FEDERATION_MODEL_ID:
        print(
            f"[WARNING] config model.name is '{model_name}' but federation requires "
            f"'{FEDERATION_MODEL_ID}'. Overriding to maintain aggregation compatibility."
        )
        model_name = FEDERATION_MODEL_ID

    device_profile, device_type = DeviceDetector.detect_hardware()
    print(f"Hardware profile : {device_profile} | Device : {device_type}")
    print(f"Loading model    : {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device_type == "cuda":
        # ── GPU path: 4-bit NF4 ───────────────────────────────────────────────
        quantization_config = None
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=quantization_type,
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            print(f"bitsandbytes unavailable ({e}), falling back to FP16 unquantized.")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    else:
        # ── CPU path: FP32, no quantization ──────────────────────────────────
        # TinyLlama-1.1B in FP32 ≈ 2.5 GB RAM — fits on any office PC.
        # device_map=None keeps all tensors on CPU without triggering
        # the accelerate device-split logic (which requires a GPU).
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
        )

    # Freeze base model — only AFLoRA B and Lambda matrices will be trained.
    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer, device_profile


# Backwards-compatible alias
load_llama = load_model
