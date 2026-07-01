# FusionNet Architecture

This document covers the high-level system architecture and component breakdown for the FusionNet federated learning system.

---

## Hardware Abstraction Layer

FusionNet uses a dynamic hardware detection framework (`hardware_utils.py`) to seamlessly support heterogeneous edge devices.

When a node joins the federated network, the framework analyses the local system:

1. **GPU Check**: Checks `torch.cuda.is_available()` to determine if an AMD ROCm or NVIDIA CUDA device is present.
2. **RAM Check**: Evaluates available system RAM (via `psutil`) or VRAM.
3. **Dynamic Scaling**: Based on the hardware tier, it returns an optimal configuration:

| Tier | Device | LoRA Rank | Batch Size | Contribution Weight |
|---|---|---|---|---|
| `MI300X` | AMD MI300X Cloud | 64 | 32 | 2.0× |
| `RX_7900_XTX` | Radeon 7900 / High-end GPU | 16–32 | 8–16 | 1.0× |
| `Steam_Deck` | Steam Deck / Low-VRAM GPU | 4–8 | 2–4 | 0.5× |
| `CPU_only` | Office PC, no GPU | 2–4 | 1–2 | 0.1× |

---

## Model Selection Constraint

> **Critical:** AFLoRA federated aggregation requires all clients in the same round to share **identical model architecture**. The coordinator averages `A` matrices by index. If Client A trains on Llama-3-8B (hidden size 4096) and Client B trains on TinyLlama-1.1B (hidden size 2048), the `A` matrix shapes do not match and aggregation crashes.

### Current Status
`config.yaml` `model.name` is set to `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` and `models/loader.py` enforces this federation-wide — any other value is overridden with a warning. All nodes run the same model regardless of hardware tier.

**Why TinyLlama:**
- Runs on CPU-only office PCs in FP32 (~2.5 GB RAM) — proves the heterogeneous pitch
- Shares the Llama transformer architecture — AFLoRA injection, target modules, and FedAvg aggregation all work with zero code changes
- Not gated — no special HuggingFace access required

### Production Path: Architecture Cohorts
In production, devices can be grouped into architecture-specific cohorts. Each cohort runs its own FedAvg round on its own global model (e.g. GPU cohort on Llama-3-8B, CPU cohort on TinyLlama). Cross-cohort knowledge transfer requires knowledge distillation — roadmap item.

---

## Dirichlet Non-IID Data Partitioning

To honestly demonstrate federated learning's value, each simulated client must hold a **genuinely different** data distribution — just as a hospital clinic holds different sentiment patterns than a bank branch.

FusionNet uses a **Dirichlet-skewed, device-tier-aware** partitioning strategy implemented in `fusionnet-client/fl_datasets/partitioner.py`.

### How It Works

For each label class `c`, the fraction of class `c`'s samples assigned to each client is drawn from a Dirichlet distribution with concentration parameter α:

```
proportions_c ~ Dirichlet([α, α, ..., α])   (length = num_clients)
```

- **High α** → proportions are nearly equal across clients → balanced, near-IID shard
- **Low α** → one client gets most of class `c` → heavy label skew, non-IID

### Device Tier → Data Configuration

Both the skewness (α) and the maximum shard size (data_fraction) are determined by the device tier, so heterogeneity is visible in the **actual data**, not just in LoRA rank numbers:

| Tier | Dirichlet α | Max Shard Size | Real-World Analogy |
|---|---|---|---|
| `MI300X` | 10.0 (balanced) | 100% of corpus | Cloud node aggregating diverse sources |
| `RX_7900_XTX` | 2.0 (mild skew) | 50% of corpus | Enterprise workstation at regional HQ |
| `Steam_Deck` | 0.5 (heavy skew) | 20% of corpus | One department within a hospital |
| `CPU_only` | 0.1 (extreme skew) | 8% of corpus | A single clinic or law firm branch |

### Partition Isolation

The `client_id` integer is added to the base random seed, so two CPU-only nodes with different IDs receive different shards even though they share the same tier configuration. This matches the real-world scenario where two law firms hold distinct legal precedents.

### Judge Visibility

At startup, each node prints a partition report automatically:

```
────────────────────────────────────────────────────────────
  DATA PARTITION REPORT  │  Tier: CPU_only
────────────────────────────────────────────────────────────
  Profile     : Highly skewed, tiny shard (CPU-only office)
  α (Dirichlet): 0.1  │  Max fraction: 8% of corpus
  Shard size  : 800 samples
  Dominant label: #42
  Label spread: 7 unique classes in shard

  Top-5 label distribution:
    Label  42: ████████████████████████████████ 79.4%
    Label   5: ███ 7.1%
    Label  18: ██ 5.3%
    Label   3: █ 4.2%
    Label  11: █ 3.0%
────────────────────────────────────────────────────────────
```

---

## Communication Layer — HF Hub Parameter Server

FusionNet uses a private Hugging Face Dataset repository (`yash-goswami/fusionnet-coordinator`) as a serverless parameter server. No custom server infrastructure is needed.

### Round protocol

```
Client                                  HF Hub repo                      Coordinator
  │                                         │                                 │
  │── train locally ──────────────────────▶ │                                 │
  │   upload round_N/client_K.pt ─────────▶ │                                 │
  │   (includes data_size metadata)         │◀── poll list_repo_files ────────│
  │                                         │    (wait for --min-clients)     │
  │                                         │── download all round_N/*.pt ───▶│
  │                                         │                                 │── Weighted FedAvg ─▶
  │                                         │◀── upload global/Global_A_N.pt ─│
  │◀── download global/Global_A_N.pt ───────│                                 │
  │    load_global_A() into AFLoRA layers   │                                 │
```

**Fault Tolerance & Weighted Aggregation:** The coordinator utilizes a timeout mechanism (`--timeout`) to prevent deadlocks if a client drops offline. It aggregates available matrices once `--min-clients` is reached. Crucially, the coordinator now performs **Data-Size Weighted FedAvg** by parsing the `data_size` metadata attached to each client's upload, ensuring that nodes with more local data exert a proportional influence on the global model.

### File layout in repo

```
fusionnet-coordinator/
├── round_1/
│   ├── client_0.pt       # local A matrices from node 0
│   ├── client_1.pt
│   └── client_2.pt
├── round_2/
│   └── ...
└── global/
    ├── Global_A_round_1.pt   # aggregated A matrices from coordinator
    └── Global_A_round_2.pt
```

### Authentication

All Hub access is authenticated via `HF_TOKEN` loaded from `.env` at startup. Both `federation/hf_hub.py` and `scripts/hf_coordinator.py` call `auth.get_token()` on init. The repo is private so gradient updates are never publicly accessible.

### Base64 serialisation (legacy path)

The original HTTP path serialised A matrices into Base64 JSON strings. This is still available in `federation/privacy.py` (`serialize_tensor_base64` / `deserialize_tensor_base64`) for any future HTTP coordinator integration, but the primary path is now `.pt` files via the HF Hub.

---

## Telemetry Backend Layer

FusionNet runs a centralized **FastAPI backend** tracking live metrics, which powers the real-time React dashboard.

### Core Architecture
- **Framework**: FastAPI (Python)
- **Database**: PostgreSQL with async SQLAlchemy and Alembic migrations (optional — see in-memory mode below).
- **Models & Schemas**: Matches exact TypeScript interfaces expected by the frontend (Device, Round, Metric, Event).
- **Real-Time Delivery**: Multi-channel WebSocket manager pushes live stats seamlessly to clients.

### Persistent Backend (PostgreSQL Required)

The backend stores all state, telemetry, and metrics in PostgreSQL.

```env
BACKEND_AUTH_DISABLED=true
```

Start the backend with:

```powershell
$env:PYTHONPATH = "."
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` is required when edge devices on other machines need to reach the backend.

### Flow & Integrations
1. **Nodes & Coordination**: Both the local training nodes (`fusionnet-client/main.py`) and the HF coordinator (`hf_coordinator.py`) silently broadcast their training status, progress, privacy budget (epsilon), and hardware health to this API.
2. **Fire-and-Forget Architecture**: Client integrations with the backend are fully asynchronous. If the backend server is down, local federated training continues uninterrupted (robust edge fault tolerance).
3. **Authentication**: Handled via `HFAuthMiddleware` by validating bearer tokens against the Hugging Face identity APIs. Set `BACKEND_AUTH_DISABLED=true` to bypass for local testing.

### Hardening & WebSockets Optimization
- **Unified WebSocket Context**: The frontend dashboard initiates a single connection to the `/ws/all` backend channel via a custom React `useWebSocket` Provider, minimizing network resource footprint. It manages auto-reconnection and parses `device.registered` and `device.heartbeat` events.
- **Max Sockets Rate-Limiting**: The backend connection manager restricts total active WebSocket connections to a maximum of 100 concurrent sockets to prevent server thread depletion.
- **Concurrent Broadcasts**: Sockets broadcast concurrently using Python's native `asyncio.gather` for optimal packet delivery speed.
- **Keep-Alive Heartbeats**: The server establishes a 30-second ping/pong validation loop on each websocket connection. Stale connections are forcefully disconnected if they do not communicate within 30 seconds.
- **Registration Input Validation**: Added strict `HardwareType` Enum validation (`MI300X`, `RX_7900_XTX`, `Steam_Deck`, `CPU_only`, plus marketing representations) on backend device registration and set default coordinates to `(50, 50)` for unrecognized client regions.

---

## Privacy Layer

FusionNet implements DP-SGD with a dual-engine approach:

1. **Primary**: Opacus `PrivacyEngine` — production-grade per-sample gradient clipping and Gaussian noise addition. After `optimizer.step()`, `optimizer.zero_grad()` is called explicitly so Opacus gradient hooks don't accumulate across batches.
2. **Fallback**: Custom `CustomPrivacyEngine` — same interface, used when Opacus hooks fail on dynamically quantized 4-bit modules (common on ROCm backends). Calls `clip → noise → step → zero_grad` in sequence.

Both engines guarantee (ε, δ)-differential privacy with `ε ≤ 1.0` and `δ ≤ 1e-5` as default targets.

### Privacy Budget Accountant
FusionNet includes a robust `PrivacyAccountant` (`federation/privacy.py`) that strictly tracks the cumulative privacy budget (epsilon, ε) consumed across multiple federated rounds.
- The accountant persists the total spent budget to disk (`privacy_state.json`), ensuring it survives client restarts.
- Before each round begins, the system verifies if there is sufficient budget remaining.
- If the total privacy cap (e.g., `target_epsilon: 10.0`) is exhausted, the accountant throws a `PrivacyBudgetExceeded` exception, immediately halting local training to preserve mathematical privacy guarantees.

---

## Configuration & Stability

To ensure stability across distributed edge nodes, FusionNet employs strict **Pydantic Configuration Validation** (`config_schema.py`). 
At startup, `client.py` validates the `config.yaml` file against the schema, verifying types, allowed values (e.g., `quantization_type in {"nf4", "none"}`), and ranges (e.g., `lora_rank >= 1`). This fail-fast mechanism prevents nodes from crashing mid-training due to misconfiguration. Furthermore, all core client operations utilize structured `logging` instead of `print()` statements for easier monitoring and parsing.

---

## AFLoRA Federated Parameter Contract
AFLoRA uses the adapter math `Delta W = A x Lambda x B`.

Current training contract:

1. **Local training updates A, B, and Lambda**: all three AFLoRA parameters are trainable on each client.
2. **Only A is federated**: after local training, the client exports/uploads the ordered `A` matrices for aggregation. This ensures the exported `A` matrices carry the client learning signal to the coordinator.
3. **B and Lambda stay local**: the client saves `B` and `Lambda` to `local_B.pt` and `local_lambda.pt`; they are never uploaded or averaged.
4. **Coordinator aggregates A only**: FedAvg runs layer-by-layer across matching client `A` matrices, then publishes the next global A.

This gives the system a shared global update through `A` while keeping local personalization in `B` and `Lambda`.

Adapter injection also correctly targets and replaces linear layers within PyTorch `nn.ModuleList` and `nn.ModuleDict` modules by executing key/index-based assignments (`model[int(name)] = aflora_layer`) instead of `setattr`.
