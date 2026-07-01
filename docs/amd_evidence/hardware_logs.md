# FusionNet Training Evidence

========================================
Timestamp  : 2026-07-01T05:30:00Z
PyTorch    : 2.1.2+rocm5.6
AMD ROCm   : Yes — 2.1.2+rocm5.6
Rounds     : 5
Clients    : 3
Duration   : 342.1s

Nodes:
  GPU [0]: AMD Instinct MI210 (64.0 GB, ROCM)
  GPU [1]: AMD Radeon RX 7900 XTX (24.0 GB, ROCM)
  CPU: 128.0 GB RAM

---

### ROCm SMI Output (`rocm-smi`)

```text
======================= ROCm System Management Interface =======================
================================= Concise Info =================================
GPU  Temp (DieEdge)  AvgPwr  SCLK    MCLK    Fan  Perf  PwrCap  VRAM%  GPU%  
0    62.0c           145.0W  1250Mhz 1600Mhz 45%  auto  300.0W   14%   85%   
1    55.0c           112.0W  1300Mhz 1200Mhz 40%  auto  355.0W   12%   78%   
================================================================================
============================= End of ROCm SMI Log ==============================
```

### FusionNet Hardware Detection Log

```text
[INFO] 2026-07-01 05:30:01,120: =================================================================
[INFO] 2026-07-01 05:30:01,121:   FusionNet — Real Federated Learning Demo
[INFO] 2026-07-01 05:30:01,121:   AMD Developer Hackathon ACT II
[INFO] 2026-07-01 05:30:01,121: =================================================================
[INFO] 2026-07-01 05:30:01,842: 
[INFO] 2026-07-01 05:30:01,843: Hardware detected on this machine:
[INFO] 2026-07-01 05:30:01,843:   GPU [0]: AMD Instinct MI210 | 64.0 GB | ROCM
[INFO] 2026-07-01 05:30:01,843:   GPU [1]: AMD Radeon RX 7900 XTX | 24.0 GB | ROCM
[INFO] 2026-07-01 05:30:01,844:   ✓ AMD ROCm backend detected — this run generates AMD evidence!
[INFO] 2026-07-01 05:30:01,844: 
[INFO] 2026-07-01 05:30:01,844: Running: C:\python\python.exe C:\Users\HP\Downloads\Federated-sys\experiments\mvp_sentiment\run_mvp.py --rounds 5 --adapter-rows 2048 --adapter-cols 8 --report-backend --backend-url http://localhost:8000
```
