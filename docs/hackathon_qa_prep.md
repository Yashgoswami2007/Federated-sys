# FusionNet: Hackathon Q&A Prep

This document contains anticipated questions from hackathon judges regarding the technical trade-offs of the FusionNet architecture, along with robust answers to defend the design.

---

## Q1: If you prune 40% of the model for weak edge devices, doesn't that cause unacceptable loss degradation?

**Answer:**
Yes, aggressive pruning does increase the local loss for that specific edge device — it becomes a much "weaker learner." However, this does not ruin the global model due to two architectural safeguards:

1. **Federated Dropout (Sub-Model Training):**
   Even though the weak device is missing 40% of the model, the 60% it *does* have is still calculating gradients based on its unique, local, isolated data. The gradients it generates are "noisy" but point in the generally correct mathematical direction.

2. **Weighted Aggregation (The Safety Net):**
   When the central server averages all the globally shared AFLoRA `A` matrices together using FedAvg, it does not treat them equally. The system uses a **Contribution Weight** multiplier. An MI300X Cloud GPU gets a `2.0×` multiplier, while a CPU-only PC gets a `0.1×` multiplier.

**Conclusion:** Because the pruned model's weight is discounted so heavily, its higher loss doesn't "pollute" the high-quality learning done by the big GPUs. It acts as a tiny, helpful nudge to the global model.

---

## Q2: If the pruned models are so weak, why bother including them in the network at all?

**Answer:**
Because in industries with extreme data gravity and privacy laws (like Healthcare or Legal), that weak CPU-only laptop might hold the *only* copy of a rare patient disease record or a specific legal precedent.

Even a highly degraded, `0.1×`-weighted mathematical nudge derived from that rare, inaccessible data is incredibly valuable for the global model to learn from. Excluding those devices means excluding the data entirely, which defeats the purpose of Federated Learning.

---

## Q3: Why is your demo using TinyLlama instead of Llama-3-8B?

**Answer:**
This is a deliberate architectural decision, not a limitation — and it actually *proves* the core pitch more honestly.

The pitch is: *"Any device, any office, participates. Data never leaves."* Llama-3-8B requires ≈6.5 GB VRAM in 4-bit or ≈32 GB RAM in FP32. A standard office PC with no GPU cannot run it — so a Llama-3-8B demo that only works on GPU nodes **contradicts the pitch** rather than proving it.

TinyLlama-1.1B runs in ≈2.5 GB RAM in FP32, participates fully on any office laptop, and shares the same Llama transformer architecture — meaning all AFLoRA injection, aggregation, and privacy code is identical. The convergence curves are real. The privacy guarantees are real. The system is proven.

For production deployments on GPU cohorts, the same system scales directly to Llama-3-8B or Phi-3 with no code changes.

---

## Q4: Can different clients run different models (e.g. Llama-3-8B on GPU, TinyLlama on CPU)?

**Answer:**
Not within the same federation round. AFLoRA's federated aggregation averages `A` matrices by index. Llama-3-8B has a hidden size of 4096 so its `A` matrix is shape `[4096, rank]`. TinyLlama has a hidden size of 2048 so its `A` matrix is shape `[2048, rank]`. These shapes do not match — aggregation would crash.

The correct production path is **architecture cohorts**: GPU nodes form one federation (all running Llama-3-8B), CPU nodes form a separate federation (all running TinyLlama). Each cohort has its own global model and runs independent FedAvg rounds. Cross-cohort knowledge transfer requires knowledge distillation, which is a roadmap item.

---

## Q5: How do you prove the data is actually heterogeneous across clients? Couldn't you have just given each client the same dataset?

**Answer:**
FusionNet uses **Dirichlet Non-IID partitioning** (`fusionnet-client/datasets/partitioner.py`) where both the shard size and label skewness are determined by the device hardware tier:

| Tier | Dirichlet α | Shard Size | Analogy |
|---|---|---|---|
| `MI300X` | 10.0 (near-IID) | 100% | Cloud node, diverse sources |
| `RX_7900_XTX` | 2.0 | 50% | Enterprise regional HQ |
| `Steam_Deck` | 0.5 | 20% | Hospital department |
| `CPU_only` | 0.1 (extreme skew) | 8% | Single clinic or law branch |

This means a CPU-only office node receives a tiny, highly skewed shard where one label dominates — exactly how a real niche specialist's data looks. When the system starts, every node prints an ASCII label distribution chart showing the skew, which judges can read directly in the terminal output without any extra tooling.

The `client_id` seeds the random partition differently for each node, so even two CPU-only machines hold distinct shards — mirroring two different law firms holding different case types.

---

## Q6: How does the global model improve if each client only sees a tiny, skewed shard?

**Answer:**
This is the core value proposition of federated learning. Each client's skewed shard is a limitation *locally* — but globally, those skews *complement each other*. The clinic that sees 80% negative sentiment and 5% positive sentiment provides a gradient signal that the cloud node (which sees everything evenly) would wash out on its own.

The weighted FedAvg aggregation combines these complementary signals: the cloud node's balanced update dominates (2.0× weight), but the clinic's rare-class signal nudges the global model toward better generalisation on tail distributions. This is measurably better than training on the cloud node's data alone — which is what you would do if the clinic's data couldn't leave the building.

---

## Q7: What prevents a malicious client from submitting poisoned gradient updates?

**Answer:**
Two mechanisms:

1. **Contribution weighting:** A CPU-only node has a `0.1×` weight. Even if its update is entirely adversarial, its influence on the global model is negligible compared to GPU nodes.
2. **Byzantine-robust aggregation (roadmap):** Production FusionNet will replace FedAvg with Krum or Bulyan aggregation, which mathematically detects and discards statistical outliers before averaging. This is documented in the risk register in `README.md`.

---

## Q8: What is your actual privacy guarantee in numbers?

**Answer:**
FusionNet targets **(ε = 1.0, δ = 1e-5)-differential privacy** per training round using Opacus DP-SGD with gradient clipping norm `C = 1.0` and Gaussian noise calibrated as:

```
σ = (C × √(2 × ln(1.25/δ))) / ε  ≈  3.05
```

The privacy budget is tracked per round using the Opacus RDP accountant. If Opacus hooks fail on a 4-bit quantized module (common on ROCm), the custom `CustomPrivacyEngine` fallback applies identical noise with the same formula, maintaining the guarantee. The ε value after 20 rounds is logged and shown in the demo output.

---

## Q9: How does your coordinator work without a dedicated server?

**Answer:**
FusionNet uses a private Hugging Face Dataset repository (`yash-goswami/fusionnet-coordinator`) as a zero-infrastructure parameter server. Clients push their local `A` matrix updates as `.pt` files to `round_N/client_K.pt`. The coordinator script polls `list_repo_files()` until all expected clients have uploaded, downloads them, runs FedAvg, and pushes the averaged result to `global/Global_A_round_N.pt`. Clients then pull that file at the start of the next round.

This means the entire federation runs with no cloud VM, no open port, and no custom API — just Python scripts and a private HF repo. It's fully auditable (every round's updates are versioned on the Hub) and costs nothing to operate beyond HF's free tier storage.

---

## Q10: How do you authenticate clients to the parameter server?

**Answer:**
Each node loads a `HF_TOKEN` from a local `.env` file (gitignored, never committed). The token is passed directly to `HfApi(token=...)` in `federation/hf_hub.py`. Since the HF dataset repo is private, only tokens with read/write access to `yash-goswami/fusionnet-coordinator` can push or pull updates. In production, each client would have a scoped write token granting access only to their own `round_N/client_K.pt` path.
