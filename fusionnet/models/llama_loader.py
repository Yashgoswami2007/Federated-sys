import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Federation model constant — must match config.yaml across ALL nodes.
FEDERATION_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# TinyLlama uses the same projection layer names as Llama-2/3.
# All AFLoRA injection and aggregation code works without modification.
TINYLLAMA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_model(model_id_or_path=None, lora_rank=8, lora_alpha=16, lora_dropout=0.05):
    """
    Loads the federation model (TinyLlama-1.1B-Chat) with hardware-appropriate
    precision and injects a LoRA adapter.

    GPU nodes  : load in 4-bit NF4 (bitsandbytes) — ~1.2 GB VRAM
    CPU nodes  : load in FP32 (no quantization)   — ~2.5 GB RAM

    Args:
        model_id_or_path (str): HuggingFace model ID or local path.
                                Defaults to FEDERATION_MODEL_ID if None.
        lora_rank (int)       : Rank of the LoRA matrices.
        lora_alpha (int)      : Alpha parameter for LoRA scaling.
        lora_dropout (float)  : Dropout probability for LoRA layers.

    Returns:
        peft_model  (PeftModel)           : Model with LoRA adapters injected.
        tokenizer   (PreTrainedTokenizer) : Associated tokenizer.
    """
    if model_id_or_path is None:
        model_id_or_path = FEDERATION_MODEL_ID

    gpu_available = torch.cuda.is_available()

    print(f"Loading tokenizer from {model_id_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if gpu_available:
        # ── GPU path: 4-bit NF4 quantization ─────────────────────────────────
        print(f"GPU detected — loading {model_id_or_path} in 4-bit NF4...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            quantization_config=quantization_config,
            device_map="auto",  # maps to ROCm or CUDA automatically
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # ── CPU path: FP32, no quantization ──────────────────────────────────
        # TinyLlama-1.1B in FP32 ≈ 2.5 GB RAM — fits on any standard office PC.
        print(f"No GPU detected — loading {model_id_or_path} in FP32 on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float32,
            device_map=None,  # explicit CPU, no device_map needed
        )
        # Freeze base weights manually (prepare_model_for_kbit_training is GPU-only)
        for param in model.parameters():
            param.requires_grad = False

    print(f"Injecting LoRA adapter (r={lora_rank}, alpha={lora_alpha})...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=TINYLLAMA_TARGET_MODULES,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model, tokenizer


# Backwards-compatible alias for any existing code that calls load_llama_4bit()
load_llama_4bit = load_model
