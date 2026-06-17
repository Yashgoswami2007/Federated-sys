# FusionNet Local Client

This is the local client component of FusionNet, a privacy-preserving federated learning system optimised for AMD hardware.

## Architecture

- **Frozen Base Model**: A base LLM loaded for the detected hardware tier. Base weights remain frozen — no full fine-tuning.
- **AFLoRA Adapter**: Splits weight updates into `ΔW = A × Λ × B`, keeping `B` and `Λ` strictly on-device for personalisation. Only `A` is sent to the coordinator.
- **Hardware-Aware**: Automatically detects the environment (MI300X, RX 7900 XTX, Steam Deck, CPU) and scales adapter rank, batch size, and precision accordingly.
- **Dirichlet Non-IID Partitioning**: Each node receives a data shard whose size and label skewness are both determined by its hardware tier, mirroring real-world data gravity (a clinic's skewed records vs. a cloud node's balanced corpus).
- **Differential Privacy**: Opacus-powered DP-SGD with a resilient custom fallback for 4-bit quantized modules.
- **Communication**: AFLoRA `A` matrices are serialised as `.pt` files and exchanged via a private Hugging Face Dataset repo (`yash-goswami/fusionnet-coordinator`) — no custom server needed.

## Quickstart

### Prerequisites: Hugging Face Authentication

This project loads models from Hugging Face. Before running, set up your token:

1. Get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read scope is sufficient)
2. Create a `.env` file in the repo root:
   ```
   HF_TOKEN=your_token_here
   ```
3. Authenticate at startup:
   ```bash
   python fusionnet-client/auth.py
   ```

The `.env` file is gitignored and will not be committed.

### Linux / macOS

```bash
cd fusionnet-client
pip install -r requirements.txt
python ../auth.py  # authenticate with Hugging Face
```

### Windows (PowerShell)

```powershell
# Option A — One-shot setup (recommended, run from repo root)
.\scripts\setup_env.ps1              # CPU (any PC)
.\scripts\setup_env.ps1 -Backend cuda  # NVIDIA GPU
.\scripts\setup_env.ps1 -Backend rocm  # AMD GPU (installs CPU build; GPU needs WSL2)

# Option B — Manual setup inside fusionnet-client\
cd fusionnet-client
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Note for AMD GPU on Windows:** PyTorch ROCm wheels are Linux-only.
> For full AMD GPU acceleration, use WSL2 (Ubuntu 22.04) and run `setup_rocm.sh` inside it.
> CPU mode works natively on Windows with no extra steps.

### Run a Local Node

```bash
# Linux / macOS
# Node 0 of a 4-client federation, running 3 FL rounds
python main.py --client-id 0 --num-clients 4 --rounds 3

# Node 1 of the same federation (gets a different Dirichlet shard)
python main.py --client-id 1 --num-clients 4 --rounds 3
```

```powershell
# Windows PowerShell (same arguments)
python main.py --client-id 0 --num-clients 4 --rounds 3
python main.py --client-id 1 --num-clients 4 --rounds 3
```

| Argument | Default | Description |
|---|---|---|
| `--client-id` | `0` | Unique integer ID for this node. Seeds the Dirichlet partition so each node gets a distinct shard. Also used as the upload key (`client_0.pt`) on the HF Hub. |
| `--num-clients` | `10` | Total number of clients in the federation. Used to split label-class proportions via Dirichlet. |
| `--rounds` | `1` | Number of FL rounds to run. Each round: pull global A → train locally → push local A to HF Hub. |

### Run the Coordinator

In a separate terminal, start the coordinator to aggregate client updates:

```bash
python scripts/hf_coordinator.py --num-clients 4 --rounds 3
```

The coordinator polls the private HF repo (`yash-goswami/fusionnet-coordinator`) until all clients have uploaded for each round, then runs FedAvg and pushes the global A matrices back.

### Run Examples

```bash
python scripts/example_train.py --client-id 0 --num-clients 4
python scripts/example_federated_round.py
```

## Data Heterogeneity (Dirichlet Non-IID)

At startup, each node prints its data partition report:

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
    ...
────────────────────────────────────────────────────────────
```

The shard configuration per tier:

| Tier | Dirichlet α | Max Shard | Simulates |
|---|---|---|---|
| `MI300X` | 10.0 (balanced) | 100% | Cloud aggregator |
| `RX_7900_XTX` | 2.0 | 50% | Enterprise HQ |
| `Steam_Deck` | 0.5 | 20% | Hospital department |
| `CPU_only` | 0.1 (extreme skew) | 8% | Single clinic / law firm |

## Structure

```
fusionnet-client/
├── main.py                    # CLI entry point (--client-id, --num-clients)
├── client.py                  # FusionNetClient orchestrator
├── auth.py                    # Hugging Face authentication (loads HF_TOKEN from .env)
├── config.yaml                # Device & training config
├── requirements.txt
├── .env                       # (gitignored) HF_TOKEN=your_token_here
├── models/
│   └── loader.py              # Hardware detection + model loading
├── aflora/
│   ├── layer.py               # AFLoRA (A × Λ × B) module
│   └── injection.py           # Target module replacer
├── federation/
│   ├── client.py              # HF Hub comms & adapter state management
│   ├── hf_hub.py              # HFParameterServer (upload/download A matrices)
│   └── privacy.py             # Abstract DP engine (Opacus + fallback)
├── training/
│   └── engine.py              # Local training loop
├── datasets/
│   ├── __init__.py
│   ├── loader.py              # Dataset loading + Dirichlet partition call
│   └── partitioner.py         # Dirichlet Non-IID partitioner (NEW)
└── scripts/
    ├── example_train.py
    └── example_federated_round.py
```
