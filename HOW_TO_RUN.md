# FusionNet — How to Run

This guide covers running FusionNet end-to-end: backend, dashboard, coordinator, and edge clients — including auto-discovery of the coordinator on your local network (WiFi or LAN).

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | [python.org/downloads](https://www.python.org/downloads/) — check "Add to PATH" |
| Node.js 18+ | [nodejs.org](https://nodejs.org/) — for the frontend dashboard |
| Git | To clone the repo |
| Hugging Face account | Free — get a **write-scope token** at [hf.co/settings/tokens](https://huggingface.co/settings/tokens) |

> PostgreSQL **is required**. The backend relies on a persistent database.

---

## Step 1 — Clone & configure

```powershell
git clone <repo-url>
cd Federated-sys
```

Create a `.env` file in the repo root:

```env
HF_TOKEN=hf_your_token_here
HF_REPO_ID=yash-goswami/fusionnet-coordinator
BACKEND_AUTH_DISABLED=true
```

---

## Step 2 — Python environment

Run once from the repo root:

```powershell
# CPU-only (any PC)
.\scripts\setup_env.ps1

# NVIDIA GPU
.\scripts\setup_env.ps1 -Backend cuda

# AMD GPU (installs CPU build on Windows; use WSL2 for GPU)
.\scripts\setup_env.ps1 -Backend rocm
```

Then authenticate with Hugging Face:

```powershell
.\venv\Scripts\Activate.ps1
python fusionnet-client/auth.py
```

---

## Step 3 — Frontend dependencies (one-time)

```powershell
cd "front end"
npm install
cd ..
```

---

## Running the System

Open **4 separate terminals** from the repo root.

### Terminal 1 — Backend

```powershell
.\venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

> `--host 0.0.0.0` makes the backend reachable by other devices on your network.  
> Dashboard: open `http://localhost:3000` after starting the frontend.

### Terminal 2 — Frontend Dashboard

```powershell
cd "front end"
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Terminal 3 — Coordinator

```powershell
.\venv\Scripts\Activate.ps1
python scripts/hf_coordinator.py --num-clients 2 --rounds 1
```

The coordinator automatically **advertises itself on the local network via mDNS** so clients can find it without any IP configuration.

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--num-clients` | 3 | How many clients to wait for before aggregating |
| `--min-clients` | same as `--num-clients` | Minimum to proceed if timeout is reached |
| `--rounds` | 1 | Number of FL rounds |
| `--timeout` | 1800 | Seconds to wait per round |
| `--port` | 8000 | Backend port to advertise via mDNS |
| `--no-advertise` | — | Disable mDNS (clients must use `--backend-url`) |

### Terminal 4 — Client(s)

```powershell
.\venv\Scripts\Activate.ps1
cd fusionnet-client
python main.py --client-id 0 --num-clients 2 --rounds 1
```

> On first run, `TinyLlama-1.1B-Chat-v1.0` (~2.5 GB) downloads from HF Hub. Cached after that.

Each device needs a unique `--client-id` (0, 1, 2, …).

---

## Auto-Discovery (Same WiFi / LAN)

When the coordinator starts, it broadcasts a **mDNS service** (`_fusionnet._tcp.local.`) on the local network. Clients on the same WiFi or LAN automatically find it — no IP configuration needed.

```
Coordinator machine  ──advertises──►  _fusionnet._tcp.local. (mDNS)
                                              │
Client machine  ◄──discovers──────────────────┘
  └─► connects to http://192.168.x.x:8000 automatically
```

### Edge device on a different machine (same network)

1. Copy the repo (or just `fusionnet-client/`) to the edge device
2. Create `.env` with your `HF_TOKEN`
3. Run setup: `.\scripts\setup_env.ps1`
4. Run the client — it finds the coordinator automatically:

```powershell
.\venv\Scripts\Activate.ps1
cd fusionnet-client
python main.py --client-id 1 --num-clients 2 --rounds 1
```

### Override discovery manually

If mDNS doesn't work on your network (some corporate/enterprise WiFi blocks it):

```powershell
python main.py --client-id 1 --num-clients 2 --backend-url http://192.168.1.42:8000
```

Or set it permanently in `fusionnet-client/config.yaml`:

```yaml
backend:
  url: "http://192.168.1.42:8000"
  enabled: true
```

### Disable discovery entirely

```powershell
python main.py --client-id 1 --no-discovery
```

---

## Simulating Multiple Clients (Single Machine)

To test federating with multiple clients on one PC:

```powershell
.\scripts\launch_fl_round.ps1 -NumClients 2 -FederationRounds 1
```

---

## Troubleshooting

**"HF_TOKEN not found"**  
→ `.env` file is missing or not in the repo root directory.

**Model download is slow**  
→ First run downloads ~2.5 GB. Cached to `~/.cache/huggingface` afterward.

**Client says "No coordinator found on LAN"**  
→ Either mDNS is blocked on your network, or the coordinator isn't running yet. Use `--backend-url http://<coordinator-ip>:8000` as a workaround.

**"Privacy budget exhausted"**  
→ Delete `fusionnet-client/checkpoints/privacy_budget.json` to reset the privacy accountant.

**Backend unreachable from another device**  
→ Make sure the backend was started with `--host 0.0.0.0` (not just `localhost`). Check your firewall allows port 8000.

**Frontend shows no data**  
→ Ensure your PostgreSQL database is running and Alembic migrations have been applied.

---

## Hardware Auto-Detection

The client detects GPU VRAM and sets LoRA rank and batch size automatically:

| Hardware | LoRA Rank | Batch Size | Contribution Weight |
|---|---|---|---|
| GPU ≥ 24 GB VRAM | 16 | 16 | 2.0 |
| GPU 16–24 GB | 8 | 4 | 1.0 |
| GPU 7.5–16 GB | 4 | 2 | 0.75 |
| GPU < 7.5 GB | 2 | 1 | 0.5 |
| CPU only | 2 | 1 | 0.1 |
