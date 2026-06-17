# FusionNet: AMD Hardware Integration

This document outlines how FusionNet leverages AMD hardware and the compatibility decisions made to support heterogeneous device deployments.

---

## ROCm and PyTorch

FusionNet uses PyTorch with ROCm support for high-performance AFLoRA fine-tuning on AMD hardware. ROCm surfaces AMD GPUs through the same `torch.cuda` API, so all device detection code (`torch.cuda.is_available()`, `torch.cuda.get_device_properties()`) works identically on ROCm and CUDA backends.

The backend is distinguished at runtime in `hardware_utils.py`:

```python
def _is_rocm() -> bool:
    return getattr(torch.version, "hip", None) is not None

backend = "rocm" if _is_rocm() else "cuda"
```

On Windows, `hardware_utils.py` adds `C:\Windows\System32` to the DLL search path before accessing `torch.cuda`, which is required for PyTorch cu128 DLL resolution.

---

## bitsandbytes for 4-bit Quantization

FusionNet uses `transformers` integrated with `bitsandbytes` to load models in `nf4` 4-bit precision with double quantization. This is the primary path for GPU nodes:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

`bitsandbytes` now natively supports ROCm 6.0, making 4-bit inference available on AMD GPUs without additional patches.

**CPU-only fallback:** On nodes with no GPU, `bitsandbytes` 4-bit quantization is skipped. The model is loaded in FP32 (or FP16 if supported) using standard `from_pretrained()` without a `quantization_config`. This is why the federation model must be a small enough architecture (e.g. TinyLlama-1.1B) to fit in CPU RAM — Llama-3-8B in FP32 requires ≈32 GB.

---

## Opacus & Custom Fallback for DP-SGD

FusionNet's privacy engine uses a dual-engine approach documented in `federation/privacy.py`:

1. **Opacus `PrivacyEngine` (primary):** Production-grade per-sample gradient clipping and calibrated Gaussian noise addition, providing mathematically sound (ε, δ)-DP guarantees.

2. **`CustomPrivacyEngine` (fallback):** Activated when Opacus hooks fail to attach to dynamically quantized modules (e.g. `bitsandbytes.nn.Linear4bit`), which is common on ROCm backends. Implements identical DP noise addition with the same formula and interface.

```python
# In federation/privacy.py
try:
    # Opacus path
    model, optimizer, dataloader, engine = privacy_engine.make_private(...)
except Exception:
    # Fallback path — same (ε, δ) guarantee, manual noise injection
    engine = CustomPrivacyEngine(model, optimizer, ...)
```

Both engines guarantee `ε ≤ 1.0, δ ≤ 1e-5` per training round.

---

## RCCL for Secure Aggregation (Planned)

RCCL (ROCm Collective Communications Library) is the AMD equivalent of NCCL and is used for multi-GPU tensor communication. PyTorch's distributed backend maps `nccl` → `rccl` automatically on ROCm, so the standard `dist.init_process_group(backend="nccl")` call works on both platforms.

Current status: `comms/rccl_backend.py` is a placeholder. The planned use is `dist.all_reduce()` for AllReduce-based gradient aggregation across nodes as part of the MPC secure aggregation layer.

---

## Custom HIP Kernel for DP Noise (Optional Optimisation)

`kernels/dp_noise.hip` contains a custom HIP kernel for constant-time Gaussian noise generation, intended as a performance optimisation over the CPU-side noise injection in `CustomPrivacyEngine`. This is only relevant on AMD ROCm hardware and is not required for correctness — the Python fallback produces identical noise distributions.

---

## Dirichlet Partitioning and AMD Compute Tiers

The data partitioning system (`datasets/partitioner.py`) is hardware-aware by design. The AMD hardware tier detected at startup directly determines both the shard size and the Dirichlet concentration parameter for the Non-IID split. This means AMD hardware capability is not just a training parameter — it shapes the data the node is allowed to see, creating a coherent heterogeneity story across compute and data layers simultaneously.

See `docs/architecture.md` for the full tier-to-partition mapping.

---

## Hugging Face Hub as Serverless Parameter Server

FusionNet uses a private HF Dataset repository (`yash-goswami/fusionnet-coordinator`) to exchange AFLoRA `A` matrices between clients and the coordinator without any custom server infrastructure.

`federation/hf_hub.py` wraps `HfApi` and `hf_hub_download` into two methods:

- `upload_local_A_matrices(round_num, client_id, updates)` — serialises a list of tensors to a `.pt` file and pushes to `round_N/client_K.pt`
- `download_global_A_matrices(round_num)` — fetches `global/Global_A_round_N.pt`, returns `None` if not yet available

Authentication is handled by `auth.get_token()`, which reads `HF_TOKEN` from `.env`. The token is passed directly to `HfApi(token=...)` so no global CLI login state is required.

The coordinator (`scripts/hf_coordinator.py`) polls `list_repo_files()` until all expected client files appear, then downloads, FedAvgs, and re-uploads the global result. This makes the coordinator fully stateless — it can be restarted at any point in a round without data loss.
