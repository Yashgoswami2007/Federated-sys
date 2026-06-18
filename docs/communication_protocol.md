# FusionNet Communication Protocol

This document is the living protocol spec for how FusionNet clients, coordinators,
and dashboards exchange federated-learning updates.

The first implementation target is a local network simulation. It proves the
round protocol before we add real transport layers such as HTTP, Hugging Face Hub,
or RCCL.

## Ownership

Primary owner: PunkMonk

Main files:

- `experiments/mvp_sentiment/run_mvp.py`
- `experiments/benchmarks/plot_convergence.py`
- `fusionnet/comms/rccl_backend.py`
- `fusionnet/core/aggregator.py`
- optional backend reporting through `backend/routers/rounds.py` and
  `backend/routers/metrics.py`

## Core Idea

Each client trains locally on private data and sends only a model update to the
coordinator. The coordinator averages client updates with FedAvg and publishes a
new global update for the next round.

Raw client data never leaves the client.

```text
client_0 update ┐
client_1 update ├──> coordinator -> FedAvg -> global_round_N update
client_2 update ┘
```

For the MVP, the "network" can be simulated locally with Python objects and
files. The protocol should still look like a real distributed system so the
transport can be swapped later.

## MVP Transport: Local Network Simulation

The first version runs from one command:

```bash
python experiments/mvp_sentiment/run_mvp.py
```

It simulates multiple clients and one coordinator in a single process.

If PyTorch is installed, the simulation uses PyTorch tensors and the existing
`fusionnet.core.aggregator.fed_avg` implementation. If PyTorch is not installed,
it falls back to NumPy arrays so the communication protocol can still be tested
immediately.

The demo must prove:

1. Multiple clients create separate weight updates.
2. The coordinator receives all expected updates for a round.
3. The coordinator averages updates using `fusionnet.core.aggregator.fed_avg`.
4. The global update is saved.
5. Metrics are saved for plotting and dashboard handoff.

## Round Lifecycle

Each federated round follows this sequence:

```text
1. coordinator starts round N
2. coordinator broadcasts current global weights to clients
3. each client trains locally
4. each client creates a ClientUpdate message
5. coordinator receives all ClientUpdate messages
6. coordinator aggregates weights with FedAvg
7. coordinator saves GlobalUpdate for round N
8. coordinator writes round metrics
9. next round starts from the new global weights
```

## ClientUpdate Message

A client update is the unit sent from a client to the coordinator.

```python
{
    "client_id": "client_0",
    "round": 1,
    "num_samples": 800,
    "hardware_tier": "CPU_only",
    "weights": {
        "adapter.weight": torch.Tensor
    },
    "metrics": {
        "loss": 1.20,
        "accuracy": 0.62,
        "epsilon": 0.30,
        "train_time_s": 4.2
    }
}
```

Required fields:

- `client_id`: stable unique client name.
- `round`: current federated round number.
- `num_samples`: number of local training samples. Used as the FedAvg weight.
- `hardware_tier`: client device class for demo visibility.
- `weights`: state-dict-like mapping of tensor names to tensors.
- `metrics`: local training metrics reported by the client.

## GlobalUpdate Message

A global update is the unit sent from the coordinator back to clients.

```python
{
    "round": 1,
    "weights": {
        "adapter.weight": torch.Tensor
    },
    "metrics": {
        "avg_loss": 1.18,
        "accuracy": 0.64,
        "clients": 3,
        "total_samples": 2400,
        "epsilon_max": 0.30
    }
}
```

Required fields:

- `round`: round number that produced these global weights.
- `weights`: aggregated model update.
- `metrics`: round-level metrics for charts and logs.

## Local Output Artifacts

The local MVP writes artifacts under:

```text
experiments/mvp_sentiment/results/
```

Expected files:

```text
metrics.json
global_round_1.pt
global_round_2.pt
global_round_3.pt
client_updates/
```

The convergence plotter reads `metrics.json` and writes:

```text
experiments/benchmarks/convergence.svg
```

`metrics.json` should be easy for `plot_convergence.py` and the frontend/backend
team to consume:

```json
[
  {
    "round": 1,
    "avg_loss": 1.18,
    "accuracy": 0.64,
    "clients": 3,
    "total_samples": 2400,
    "epsilon_max": 0.30
  }
]
```

## Initial Simulation Rules

The first demo does not need to load a real LLM.

It can simulate model updates with small PyTorch tensors:

```python
{
    "adapter.weight": torch.randn(8, 8)
}
```

Each client should produce a slightly different update and metric profile so the
demo visibly proves that multiple clients participated.

Example client tiers:

| Client | Hardware Tier | Sample Count | Contribution |
|---|---:|---:|---:|
| `client_0` | `CPU_only` | 400 | small private shard |
| `client_1` | `Steam_Deck` | 900 | medium shard |
| `client_2` | `RX_7900_XTX` | 1600 | larger shard |

FedAvg should weight each client's update by `num_samples`.

## Success Criteria

The local MVP is successful when this command:

```bash
python experiments/mvp_sentiment/run_mvp.py
```

prints a clear round log:

```text
FusionNet MVP Demo
Round 1: received 3 client updates
Round 1: aggregated global weights
Round 1: loss 1.18, accuracy 0.64, epsilon 0.30
```

and writes:

```text
experiments/mvp_sentiment/results/metrics.json
experiments/mvp_sentiment/results/global_round_1.pt
```

## Transport Roadmap

The protocol should stay stable while transport changes underneath.

Planned transport layers:

1. `local`: in-process simulation and filesystem artifacts.
2. `http`: clients POST updates to a coordinator API.
3. `hf_hub`: clients upload `.pt` updates to a private Hugging Face Dataset repo.
4. `rccl`: AMD/GPU-focused collective communication for advanced demos.

The message shape should remain close to `ClientUpdate` and `GlobalUpdate` across
all transports.

## Open Questions

- Should the MVP aggregate generic `adapter.weight` tensors or AFLoRA `A`
  matrices specifically?
- Should client updates be saved individually for judge inspection?
- Should backend reporting be optional via `--report-backend`?
- Should the local simulation run in one process first, then multiple Python
  processes second?
