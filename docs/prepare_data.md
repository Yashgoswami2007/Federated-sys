# prepare_data.py — Documentation


**Purpose:** Downloads the Banking77 dataset, partitions it across 10 simulated
clients (IID and Non-IID), and assigns realistic hardware resource profiles to
each client — matching the experimental setup in the HAFLQ paper

---

## Quick Start

```bash
pip install datasets numpy
python data/partition_data.py
```

Expected output:

```
Loading Banking77 dataset...

── IID Partition ──
  Client 0: 1000 samples (IID)
  Client 1: 1000 samples (IID)
  ...

── Non-IID Partition ──
  Client 0: 847 samples, labels: [2, 5, 11, 34, 41]...
  Client 1: 912 samples, labels: [8, 19, 23, 55, 60]...
  ...

Saved data/clients/iid/
Saved data/clients/noniid/
Saved data/dataset_summary.json
Done! Data ready for federated training.
```

---

## What This Script Produces

```
data/
├── clients/
│   ├── iid/
│   │   ├── client_0.json
│   │   ├── client_1.json
│   │   └── ... (10 files)
│   └── noniid/
│       ├── client_0.json
│       ├── client_1.json
│       └── ... (10 files)
└── dataset_summary.json
```

Each `client_X.json` file contains everything a federated client needs:
its local dataset AND its hardware resource profile. Example:

```json
{
  "client_id": "client_0",
  "split_type": "noniid",
  "resources": {
    "tier": "low",
    "lora_rank": 2,
    "freeze_ratio": 0.75,
    "distance_m": 1100,
    "max_bits_mb": 10.0
  },
  "data": {
    "texts": ["I want to check my balance", "Was my transfer sent?", "..."],
    "labels": [3, 45, 12],
    "size": 847
  }
}
```

---

## Dataset: Banking77

| Property        | Value                              |
|-----------------|------------------------------------|
| Source          | HuggingFace — `PolyAI/banking77`   |
| Task            | Text classification (intent detection) |
| Training samples | 10,003                            |
| Test samples    | 3,080                              |
| Number of labels | 77 banking intent categories      |
| Example labels  | `balance_inquiry`, `card_stolen`, `transfer_abroad` |
| Paper reference | Used directly in HAFLQ (Section VII) |

Why Banking77? The HAFLQ paper evaluated on this exact dataset, so our
simulated results are directly comparable to the paper's reported numbers.

---

## Configuration Constants

These are at the top of the file. Change them to adjust the simulation.

| Constant                  | Default | Meaning                                      |
|---------------------------|---------|----------------------------------------------|
| `NUM_CLIENTS`             | `10`    | Number of simulated federated clients        |
| `LABELS_PER_CLIENT_NONIID`| `8`     | How many of 77 labels each client sees       |
| `OUTPUT_DIR`              | `data/clients` | Where JSON files are saved          |
| `SEED`                    | `42`    | Random seed — keeps results reproducible     |

---

## IID vs Non-IID Partitioning

### IID (Independent and Identically Distributed)

Every client gets a random, equal slice of all 77 labels. Unrealistic but
used as a baseline comparison.

```
Client 0: 1000 samples — ~13 samples per label — all labels represented
Client 1: 1000 samples — ~13 samples per label — all labels represented
...
```

**How it works in code:**

```python
indices = np.random.permutation(len(data))   # shuffle all 10,003 indices
splits  = np.array_split(indices, num_clients)  # cut into 10 equal piles
```

Each pile is random so label distribution is roughly equal across clients.

---

### Non-IID (Real World Scenario)

Each client only sees data from 8 out of 77 labels. This simulates reality —
different bank branches serve different customer types.

```
Client 0: 847 samples  — only sees labels [2, 5, 11, 34, 41, 55, 63, 70]
Client 1: 912 samples  — only sees labels [8, 19, 23, 45, 55, 61, 68, 74]
Client 2: 780 samples  — only sees labels [3, 14, 27, 38, 50, 59, 65, 72]
...
```

**How it works in code:**

```python
# Step 1: Group all samples by their label
label_to_indices = defaultdict(list)
for idx, item in enumerate(data):
    label_to_indices[item["label"]].append(idx)

# Step 2: Shuffle the label list so assignment is random
all_labels = list(label_to_indices.keys())
np.random.shuffle(all_labels)

# Step 3: Give each client a sliding window of 8 labels
start = (i * labels_per_client) % len(all_labels)
assigned_labels = [all_labels[(start + j) % len(all_labels)]
                   for j in range(labels_per_client)]
```

Each client then receives the first 50% of samples for each of its
assigned labels (so labels can overlap between clients — realistic).

**Why non-IID makes federated learning harder:**

When clients train on different distributions, their model updates point
in different directions. Averaging them causes "client drift" — the global
model gets pulled in conflicting directions and converges slowly or not at all.
This is the core problem that HAFLQ and AFLoRA are designed to solve.

---

## Client Resource Profiles

This is where the federated learning research connects to the data setup.
Each client is assigned a resource profile matching the HAFLQ paper
(Section VII, Table II).

### The Three Tiers

| Tier   | Clients | LoRA Rank | Freeze Ratio | Distance      |
|--------|---------|-----------|--------------|---------------|
| Low    | 0, 1, 2 | 2         | 0.75         | 1100–1300 m   |
| Medium | 3, 4, 5 | 4         | 0.50         | 1400–1600 m   |
| High   | 6, 7, 8, 9 | 8      | 0.00         | 1700–2000 m   |

---

### Field-by-Field Explanation

#### `tier` — Compute Category

Think of this as the class of hardware:

- `"low"` — weak laptop, old GPU, very limited memory
- `"medium"` — decent desktop, mid-range GPU
- `"high"` — powerful workstation, modern GPU server

This is the root from which all other resource values derive.

---

#### `lora_rank` — LoRA Adapter Size

LoRA replaces full model fine-tuning with two small matrices B and A:

```
ΔW = B × A
```

The rank `r` controls how big these matrices are. If the original weight
matrix W is 1000×1000, then:

- Rank 2 → B is 1000×2, A is 2×1000 → 4,000 trainable parameters
- Rank 4 → B is 1000×4, A is 4×1000 → 8,000 trainable parameters
- Rank 8 → B is 1000×8, A is 8×1000 → 16,000 trainable parameters

Higher rank = more expressive = better accuracy = more compute required.

**Why different ranks across clients?**

Forcing all clients to use rank 2 (to match the weakest) wastes the
potential of powerful clients. Our system lets each client use the rank
appropriate for its hardware — this is the heterogeneity problem that
HETLoRA, HAFLQ, and AFLoRA all address.

| Client Tier | LoRA Rank | Parameters Trained |
|-------------|-----------|-------------------|
| Low         | 2         | ~4,000            |
| Medium      | 4         | ~8,000            |
| High        | 8         | ~16,000           |

---

#### `freeze_ratio` — Fraction of Parameters Frozen

This comes directly from the **importance-based parameter freezing scheme**
in HAFLQ (Section IV-C).

With rank 8, there are 8 "rank-1 matrices" (components) in the LoRA adapter.
A weak client cannot compute gradients for all 8. Two options:

**Option A — Truncation (bad):** Give the weak client only 2 rank-1 matrices.
The global model loses information about the other 6 dimensions permanently.

**Option B — Freezing (good, what we use):** Give the weak client all 8
rank-1 matrices, but freeze 6 of them (lock their values). Only update
the 2 most important ones. The global model retains all 8 dimensions —
nothing is lost.

```
freeze_ratio = 0.75 → 75% of rank-1 matrices are frozen
                     → only 25% are actively trained
                     → for rank 8: freeze 6, train 2

freeze_ratio = 0.50 → freeze 4 out of 8, train 4

freeze_ratio = 0.00 → freeze nothing, train all 8
```

The server tells clients which rank-1 matrices are most important
(using importance scores) so clients always freeze the least important ones.

---

#### `distance_m` — Distance From Base Station (metres)

The HAFLQ paper models wireless uplink communication between clients
and a base station (like a 5G tower). Distance directly affects signal
quality and therefore how much data a client can transmit per round.

```
Client 0: 1100 m  ← closest, strongest signal
Client 1: 1200 m
Client 2: 1300 m
...
Client 9: 2000 m  ← furthest, weakest signal
```

This matches the exact setup stated in the paper:
> "clients are positioned at increasing distances from the base station,
> ranging from 1100 meters to 2000 meters in increments of 100 meters"

---

#### `max_bits_mb` — Maximum Uploadable Data Per Round (MB)

Derived from distance using a simplified version of the Shannon capacity
formula used in HAFLQ (Equation 11–12):

```python
max_bits_mb = round(10.0 * (2000 - distance_m) / 900, 2)
```

| Client | Distance | Max Upload |
|--------|----------|------------|
| 0      | 1100 m   | 10.00 MB   |
| 3      | 1400 m   | 6.67 MB    |
| 6      | 1700 m   | 3.33 MB    |
| 9      | 2000 m   | 0.00 MB    |

This value is used in `run_mvp.py` to simulate the importance-aware
bandwidth-adaptive quantization: clients with low bandwidth must compress
their updates more aggressively to fit within this limit.

---

## Functions Reference

### `load_banking77()`

Downloads and returns the Banking77 dataset from HuggingFace.

```python
train_data, test_data = load_banking77()
# train_data: 10,003 samples
```

No parameters. Requires internet connection on first run. Cached locally
after first download.

---

### `iid_partition(data, num_clients)`

Randomly splits data equally across clients. Each client gets
`len(data) / num_clients` samples with similar label distribution.

| Parameter    | Type    | Description                    |
|--------------|---------|--------------------------------|
| `data`       | Dataset | HuggingFace dataset object     |
| `num_clients`| int     | Number of clients to split into|

Returns `dict` — keys are `"client_0"` through `"client_9"`, values
are dicts with `texts`, `labels`, `size`.

---

### `noniid_partition(data, num_clients, labels_per_client=8)`

Splits data so each client only sees a subset of labels. Simulates
real-world data heterogeneity.

| Parameter           | Type    | Description                          |
|---------------------|---------|--------------------------------------|
| `data`              | Dataset | HuggingFace dataset object           |
| `num_clients`       | int     | Number of clients                    |
| `labels_per_client` | int     | How many labels each client receives |

Returns same format as `iid_partition`. Clients with overlapping label
assignments receive 50% of that label's samples.

---

### `assign_client_resources()`

Creates hardware resource profiles for all clients matching the HAFLQ
paper's experimental setup (Section VII).

No parameters.

Returns `dict` — keys are `"client_0"` through `"client_9"`, values
are resource profile dicts with tier, lora_rank, freeze_ratio,
distance_m, max_bits_mb.

---

### `save_client_data(client_data, resources, split_type)`

Merges data and resource profiles and saves one JSON file per client.

| Parameter     | Type   | Description                           |
|---------------|--------|---------------------------------------|
| `client_data` | dict   | Output from iid/noniid partition      |
| `resources`   | dict   | Output from assign_client_resources() |
| `split_type`  | str    | `"iid"` or `"noniid"`                |

Saves to `data/clients/{split_type}/client_{i}.json`.

---

### `save_summary(train_data, test_data, resources)`

Saves a human-readable summary of the entire data setup to
`data/dataset_summary.json`. Useful for documentation and
double-checking the setup is correct.

---

## Paper References

| Concept             | Paper         | Section        |
|---------------------|---------------|----------------|
| LoRA rank heterogeneity | HAFLQ    | Section IV     |
| Parameter freezing scheme | HAFLQ  | Section IV-C   |
| Wireless channel model | HAFLQ      | Section II-C   |
| Client distance setup | HAFLQ       | Section VII    |
| Non-IID partitioning | AFLoRA       | Section VI-A   |
| Banking77 dataset   | HAFLQ         | Section VII    |
| FedAvg baseline     | DFL Survey    | Section II-A   |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: datasets` | Library not installed | `pip install datasets` |
| `ConnectionError` | No internet | Run on a machine with internet access for first download |
| `FileNotFoundError: data/clients` | Output dir missing | Script creates it automatically — check write permissions |
| Different sample counts each run | Seed not set | `SEED = 42` is set at top — do not remove `np.random.seed(SEED)` |
| `KeyError: 'text'` | Wrong dataset field name | Banking77 uses `"text"` and `"label"` — do not rename |

---

## Dependencies

```
datasets>=2.0.0      # HuggingFace datasets library
numpy>=1.21.0        # Array operations and random shuffling
```

Install with:

```bash
pip install datasets numpy
```

---

*This file is part of the AMD Hackathon project on efficient federated
fine-tuning of LLMs. See `docs/README.md` for the full project overview.*
