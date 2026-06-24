# AMD Hardware Evidence — Collection Guide

This directory contains hardware evidence logs from running FusionNet on AMD hardware.
Judges can use these files to verify AMD Developer Cloud and ROCm usage.

## Files to Collect (run these commands on AMD hardware)

### 1. GPU Info (`rocm_smi_output.txt`)
```bash
# On AMD Developer Cloud VM or local AMD GPU:
rocm-smi --showallinfo > docs/amd_evidence/rocm_smi_output.txt
```

### 2. Backend Detection (`backend_detection.txt`)
```bash
python -c "
import torch
import sys
sys.path.insert(0, '.')
from fusionnet.core.hardware_utils import detect_hardware
cfg = detect_hardware()
print('Hardware config:', cfg)
" > docs/amd_evidence/backend_detection.txt
```
Expected output: `Backend: ROCM`

### 3. HIP Kernel Compile (`hip_compile_output.txt`)
```bash
# Requires hipcc (part of ROCm toolchain)
hipcc -O3 -DDP_NOISE_STANDALONE_TEST -o /tmp/dp_noise_test \
    fusionnet/kernels/dp_noise.hip -lhiprand 2>&1 | tee docs/amd_evidence/hip_compile_output.txt
/tmp/dp_noise_test | tee -a docs/amd_evidence/hip_compile_output.txt
```

### 4. RCCL Benchmark (`rccl_benchmark.txt`)
```bash
# Single AMD GPU:
python scripts/rccl_demo.py --rank 0 --world-size 1 2>&1 | tee docs/amd_evidence/rccl_benchmark.txt

# Two AMD GPUs (better demonstration):
python -m torch.distributed.run --nproc_per_node=2 scripts/rccl_demo.py 2>&1 \
    | tee docs/amd_evidence/rccl_benchmark.txt
```

### 5. FL Training on AMD (`amd_training_log.txt`)
```bash
cd fusionnet-client
python main.py --client-id 0 --num-clients 1 --rounds 3 2>&1 \
    | tee ../docs/amd_evidence/amd_training_log.txt
```

### 6. AMD Developer Cloud Screenshots
- Provision a VM and screenshot the console showing instance type (MI210/MI300X)
- Save as `docs/amd_evidence/amd_cloud_console.png`

---

## AMD Developer Cloud Setup (for judges)

```bash
# 1. SSH into AMD Dev Cloud VM
# 2. Verify ROCm:
rocm-smi
hipcc --version

# 3. Clone and setup FusionNet:
git clone https://github.com/YOUR_USERNAME/Federated-sys.git
cd Federated-sys
bash scripts/setup_rocm.sh

# 4. Run the demo:
python scripts/run_real_fl_demo.py --rounds 5 --num-clients 3
```

---

## Expected Output on AMD Hardware

```
Backend    : AMD ROCm 6.1.0
PyTorch    : 2.3.0+rocm6.1
GPU [0]    : AMD Instinct MI300X | VRAM: 192.0 GB
```
or on local AMD 4GB GPU (Node 2):
```
Backend    : AMD ROCm 6.0.0
PyTorch    : 2.3.0+rocm6.0
GPU [0]    : AMD Radeon RX 6600 | VRAM: 8.0 GB
```
