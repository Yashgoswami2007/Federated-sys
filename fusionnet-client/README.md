# FusionNet Local Client

This is the local client component of FusionNet, a privacy-preserving federated learning system optimised for AMD hardware.

## Architecture

- **Frozen Base Model**: A base LLM loaded for the detected hardware tier. Base weights remain frozen — no full fine-tuning.
- **AFLoRA Adapter**: Splits weight updates into `ΔW = A × Λ × B`, keeping `B` and `Λ` strictly on-device for personalisation. Only `A` is sent to the coordinator.
- **Hardware-Aware**: Automatically detects the environment (MI300X, RX 7900 XTX, Steam Deck, CPU) and scales adapter rank, batch size, and precision accordingly.
- **Dirichlet Non-IID Partitioning**: Each node receives a data shard whose size and label skewness are both determined by its hardware tier, mirroring real-world data gravity (a clinic's skewed records vs. a cloud node's balanced corpus).
- **Differential Privacy**: Opacus-powered DP-SGD with a resilient custom fallback for 4-bit quantized modules.
- **Communication**: Base64 serialisation of `A` matrices for JSON-safe HTTP communication.

## Quickstart

```bash
cd fusionnet-client
pip install -r requirements.txt
```

### Run a Local Node

```bash
# Node 0 of a 4-client federation (auto-detects hardware tier)
python main.py --client-id 0 --num-clients 4

# Node 1 of the same federation (gets a different Dirichlet shard)
python main.py --client-id 1 --num-clients 4
```

| Argument | Default | Description |
|---|---|---|
| `--client-id` | `0` | Unique integer ID for this node. Seeds the Dirichlet partition so each node gets a distinct shard. |
| `--num-clients` | `10` | Total number of clients in the federation. Used to split label-class proportions via Dirichlet. |

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
├── config.yaml                # Device & training config
├── requirements.txt
├── models/
│   └── loader.py              # Hardware detection + model loading
├── aflora/
│   ├── layer.py               # AFLoRA (A × Λ × B) module
│   └── injection.py           # Target module replacer
├── federation/
│   ├── client.py              # Base64 comms & state management
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
