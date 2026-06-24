# Zero Raw Data Transmission — Network Privacy Proof

This document describes how to verify that FusionNet transmits **zero bytes of raw training data** between federated nodes during a training round.

---

## What Gets Transmitted

FusionNet's HF Hub communication layer transmits **only serialized PyTorch tensors** (`.pt` files):

| File path in HF repo | Contents | Size (rank=8, TinyLlama) |
|---|---|---|
| `round_N/client_K.pt` | AFLoRA A matrices for all injected layers | ~700 KB |
| `global/Global_A_round_N.pt` | Aggregated global A matrices | ~700 KB |

No text, no labels, no feature vectors, no raw gradients (only DP-noised, clipped LoRA adapter matrices).

---

## Verification Method 1: HF Hub Audit Log

Since every file upload to the private HF Hub repository is versioned and logged, an auditor can:

1. Download the full commit history of `yash-goswami/fusionnet-coordinator`
2. Verify that all uploaded files are `.pt` tensor files (not `.txt`, `.csv`, `.json` with text)
3. Inspect a sample `.pt` file — it contains only floating-point tensors, no text

```python
import torch
payload = torch.load("round_1/client_0.pt", weights_only=True)
# payload = {"matrices": [tensor([...]), ...], "data_size": 800}
# No strings, no raw text, no labels — only float tensors
print(type(payload["matrices"][0]))   # <class 'torch.Tensor'>
```

---

## Verification Method 2: Network Packet Capture

Run a real FL round while capturing all network traffic:

```bash
# On Windows (requires Npcap):
# Start Wireshark, filter: tcp.port == 443 or tcp.port == 80
# Run the FL client:
python fusionnet-client/main.py --client-id 0 --num-clients 2 --rounds 1

# What you will see:
#  - HTTPS connections ONLY to huggingface.co (port 443)
#  - No connections to any client IP addresses
#  - No connections to any external AI API endpoints
#  - Payload is TLS-encrypted (HTTPS), but file type is .pt (tensor)
```

On Linux with tcpdump:
```bash
sudo tcpdump -i any -w /tmp/fusionnet_capture.pcap port 443 &
python fusionnet-client/main.py --client-id 0 --num-clients 2 --rounds 1
sudo kill %1

# Analyze:
tcpdump -r /tmp/fusionnet_capture.pcap | grep -v huggingface | head -20
# Should show ZERO connections to non-HF endpoints
```

---

## Verification Method 3: Dataset Isolation Check

The Dirichlet partitioner assigns each client a private shard:

```python
from fusionnet_client.fl_datasets.partitioner import dirichlet_partition
from datasets import load_dataset

ds = load_dataset("banking77")["train"]
# Client 0 and Client 1 get non-overlapping index sets
shard_0 = dirichlet_partition(ds, device_tier="CPU_only", client_id=0, num_clients=3)
shard_1 = dirichlet_partition(ds, device_tier="CPU_only", client_id=1, num_clients=3)

overlap = set(shard_0.indices) & set(shard_1.indices)
print(f"Data overlap: {len(overlap)} samples")  # Should be 0
```

---

## What the Commitment Scheme Adds

With `fusionnet/core/zkp_verify.py`, the coordinator can verify **without seeing the raw gradient**:

1. Client computes: `commit(ΔW_clipped, nonce)` → sends commitment hash
2. Client trains locally and uploads `ΔW_clipped`
3. Client reveals `nonce` after upload
4. Coordinator verifies: `SHA256(ΔW_clipped || nonce) == commitment` AND `||ΔW_clipped||₂ ≤ C`

This proves the gradient was not altered between commit and upload — a cryptographic integrity check without additional data leakage.

---

## Summary

| Claim | Verification | Status |
|---|---|---|
| Raw text never transmitted | HF Hub audit / packet capture | ✅ By design |
| Only tensor files cross the wire | Inspect `.pt` file contents | ✅ Verifiable |
| Clients don't share training data | Dirichlet partition index check | ✅ Code-verifiable |
| Updates satisfy DP noise | Opacus epsilon accounting | ✅ Logged per round |
| Updates not tampered post-commit | ZKP commitment verification | ✅ `zkp_verify.py` |
