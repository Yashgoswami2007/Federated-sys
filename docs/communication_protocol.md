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
- `fusionnet/comms/local_backend.py`
- `fusionnet/comms/http_event_sink.py`
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

## Current Network Topology

Current MVP topology:

```text
simulated edge clients
  -> LocalCommunicationBackend
  -> FedAvg coordinator logic
  -> HttpEventSink
  -> FastAPI backend
  -> REST/WebSocket data for dashboard
```

In this mode, all clients run inside one Python process, but the messages are
shaped like real VM-to-coordinator communication.

Target topology:

```text
edge VMs
  -> upload/download model updates through HF Hub or another weight transport
  -> report round status, metrics, and events to FastAPI
  -> frontend reads FastAPI REST endpoints and WebSocket events
```

The important split is:

```text
weights move through the comms/coordinator path.
status and metrics move through the backend telemetry path.
frontend only reads backend APIs.
```

## MVP Transport: Local Network Simulation

The first version runs from one command:

```bash
python experiments/mvp_sentiment/run_mvp.py
```

It simulates multiple clients and one coordinator in a single process.

The reusable communication layer lives in
`fusionnet/comms/local_backend.py`. The demo script uses it as a local
parameter server:

```python
comms.start_round(round_num, global_weights)
comms.submit_update(client_update)
updates = comms.wait_for_updates(round_num)
global_weights = comms.aggregate(round_num)
comms.publish_global_update(round_num, global_weights, round_metrics)
comms.write_metrics(metrics)
```

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

## Backend Event Hooks

The communication backend accepts an optional event sink. This keeps telemetry
separate from training: if the backend is offline, the coordinator can still run.

The HTTP event sink lives in `fusionnet/comms/http_event_sink.py`.

Run local simulation only:

```bash
python experiments/mvp_sentiment/run_mvp.py
```

Run local simulation and attempt backend reporting:

```bash
python experiments/mvp_sentiment/run_mvp.py --report-backend
```

Optional backend arguments:

```bash
python experiments/mvp_sentiment/run_mvp.py \
  --report-backend \
  --backend-url http://localhost:8000 \
  --backend-token "$HF_TOKEN"
```

All backend calls are best-effort. A failed POST/PATCH is logged in verbose mode
but does not stop local training, aggregation, metrics, or artifact writing.

For local MVP testing, ensure PostgreSQL is configured and run the backend:

```powershell
# Windows (PowerShell)
$env:BACKEND_AUTH_DISABLED = "true"
$env:PYTHONPATH = "."
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
# Linux / WSL2
BACKEND_AUTH_DISABLED=true \
  python -m uvicorn backend.main:app --reload --port 8000
```

### Verified MVP Backend Flow

This flow was verified locally.

Terminal 1: start the FastAPI backend (Windows):

```powershell
.\venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
$env:BACKEND_AUTH_DISABLED = "true"
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2: run the local federated round and report events to the backend:

```powershell
.\venv\Scripts\Activate.ps1
python experiments/mvp_sentiment/run_mvp.py --rounds 1 --report-backend
```

Success signal:

```text
Backend reporting enabled: http://localhost:8000
Round 1: received 3 client updates
Round 1: aggregated global weights with FedAvg
Saved metrics to experiments/mvp_sentiment/results/metrics.json
Saved final global weights to experiments/mvp_sentiment/results/global_round_1.pt
```

There should be no `[backend-report] ... failed` lines.

Verify that the backend received the comms events:

```bash
curl http://localhost:8000/api/rounds/jobs
curl http://localhost:8000/api/events/activity
curl http://localhost:8000/api/metrics/loss-curve
curl http://localhost:8000/api/dashboard/kpi
```

Expected meaning:

| Endpoint | What it proves |
|---|---|
| `/api/rounds/jobs` | backend knows the round completed |
| `/api/events/activity` | backend received round lifecycle events |
| `/api/metrics/loss-curve` | backend received training metrics |
| `/api/dashboard/kpi` | backend can serve dashboard-ready summary data |

This verifies:

```text
local simulated clients
  -> LocalCommunicationBackend
  -> FedAvg
  -> HttpEventSink
  -> FastAPI backend route contract
  -> dashboard-ready API data
```

This does not verify persistent PostgreSQL storage. PostgreSQL remains a backend
infrastructure task.

### Backend Boundary

The communication layer should not expose frontend endpoints directly.

Responsibility split:

```text
comms:
  coordinates rounds, receives client updates, aggregates weights, emits events

backend:
  receives comms events, stores state/metrics/events, exposes REST/WebSocket APIs

frontend:
  consumes backend APIs and WebSocket events
```

Short version:

```text
comms tells backend what happened.
backend records and exposes it.
frontend displays it.
```

Local backend events:

| Event | Meaning | Future backend target |
|---|---|---|
| `round.started` | coordinator opened round N | `POST /api/rounds` |
| `client.update_received` | client update arrived | `PATCH /api/rounds/{round}` |
| `round.aggregated` | FedAvg completed | `POST /api/events` |
| `global.update_published` | global update saved | `PATCH /api/rounds/{round}` and `POST /api/models/global` |
| `metrics.written` | local metrics artifact saved | optional dashboard/debug event |

Current HTTP mappings:

| Event | Backend calls |
|---|---|
| `round.started` | `POST /api/rounds`, `POST /api/events` |
| `client.update_received` | `PATCH /api/rounds/{round}`, `POST /api/metrics` |
| `round.aggregated` | `POST /api/events` |
| `global.update_published` | `PATCH /api/rounds/{round}`, `POST /api/metrics`, `POST /api/models/global`, `POST /api/events` |
| `metrics.written` | `POST /api/events` |

The next backend step is not to change the communication protocol. It is to run
FastAPI locally and verify these event mappings against the database/dashboard.

## Round Lifecycle

Each federated round follows this sequence:

```text
1. coordinator starts round N
2. coordinator broadcasts current global weights to clients
3. each client trains locally
4. each client creates a ClientUpdate message
5. coordinator receives all ClientUpdate messages
6. coordinator validates updates (filters NaN/Inf, clips extreme norms for BFT)
7. coordinator aggregates weights with FedAvg (or FedMedian)
8. coordinator saves GlobalUpdate for round N
9. coordinator writes round metrics
10. next round starts from the new global weights
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
