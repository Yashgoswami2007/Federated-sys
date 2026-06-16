import numpy as np
from torch.utils.data import Subset

# ─────────────────────────────────────────────────────────────────────────────
# Device-Tier → Data Partition Configuration
#
# alpha  : Dirichlet concentration parameter.
#          HIGH alpha (≥5.0) → near-uniform label distribution (balanced shard)
#          LOW  alpha (≤0.3) → one or two labels dominate (skewed shard)
#          This maps naturally to the real world:
#            - A cloud GPU node aggregates data from many sources → balanced
#            - A hospital clinic specialises in one disease → heavily skewed
#
# data_fraction : Fraction of the total dataset this tier is allowed to see.
#          Strong devices handle more data; weak CPU-only devices get a small
#          shard, reflecting their real-world storage and bandwidth limits.
# ─────────────────────────────────────────────────────────────────────────────

TIER_PARTITION_CONFIG = {
    "MI300X": {
        "alpha": 10.0,           # Near-IID: large cloud node sees balanced data
        "data_fraction": 1.0,    # Full dataset access
        "description": "Balanced, full-scale (cloud aggregator node)",
    },
    "RX_7900_XTX": {
        "alpha": 2.0,            # Mildly skewed: enterprise GPU workstation
        "data_fraction": 0.50,   # 50% of dataset
        "description": "Mildly skewed, mid-scale (enterprise GPU workstation)",
    },
    "Steam_Deck": {
        "alpha": 0.5,            # Significantly skewed: consumer edge device
        "data_fraction": 0.20,   # 20% of dataset
        "description": "Skewed, small shard (consumer edge / gaming device)",
    },
    "CPU_only": {
        "alpha": 0.1,            # Highly skewed: a niche specialist (clinic, law firm)
        "data_fraction": 0.08,   # 8% of dataset — realistic for a single office
        "description": "Highly skewed, tiny shard (CPU-only office / specialist node)",
    },
}

# Default fallback if an unknown profile is detected
_DEFAULT_TIER = "CPU_only"


def dirichlet_partition(dataset, device_tier: str, client_id: int = 0, num_clients: int = 10, seed: int = 42) -> Subset:
    """
    Carves a Dirichlet-skewed, size-bounded shard out of `dataset` for one client.

    The shard size AND label skewness are both determined by `device_tier`:
      - Strong GPU tier  → large shard, balanced label distribution (high α)
      - Weak CPU tier    → small shard, heavily skewed distribution (low α)

    This means the heterogeneity story is visible in the actual data the judges
    see, not just in LoRA rank or batch-size differences.

    Args:
        dataset      : A HuggingFace dataset (must have a 'labels' column).
        device_tier  : Hardware profile string matching TIER_PARTITION_CONFIG keys.
        client_id    : Unique integer ID for this client node (used as RNG seed offset
                       so two clients of the same tier still get different shards).
        num_clients  : Total number of simulated clients in the federation.
        seed         : Base random seed for reproducibility.

    Returns:
        Subset of `dataset` representing this client's private shard.
    """
    tier_cfg = TIER_PARTITION_CONFIG.get(device_tier, TIER_PARTITION_CONFIG[_DEFAULT_TIER])
    alpha          = tier_cfg["alpha"]
    data_fraction  = tier_cfg["data_fraction"]

    rng = np.random.default_rng(seed + client_id)

    # ── 1. Extract label array ────────────────────────────────────────────────
    labels = np.array(dataset["labels"])
    unique_classes = np.unique(labels)
    num_classes    = len(unique_classes)

    # ── 2. Dirichlet assignment per class ─────────────────────────────────────
    # For each class c, draw a Dirichlet vector of length num_clients.
    # Each entry is the fraction of that class's samples assigned to that client.
    client_indices = []

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)

        # Proportions for ALL clients for this class
        proportions = rng.dirichlet([alpha] * num_clients)

        # Cumulative split points
        split_points = (np.cumsum(proportions) * len(cls_indices)).astype(int)
        split_points = np.clip(split_points, 0, len(cls_indices))

        # Extract THIS client's slice
        start = split_points[client_id - 1] if client_id > 0 else 0
        end   = split_points[client_id]
        client_indices.extend(cls_indices[start:end].tolist())

    # ── 3. Apply data-fraction cap ────────────────────────────────────────────
    max_samples = max(1, int(len(dataset) * data_fraction))
    rng.shuffle(client_indices)
    client_indices = client_indices[:max_samples]

    return Subset(dataset, client_indices)


def describe_partition(subset: Subset, dataset, num_classes: int = None) -> dict:
    """
    Returns a summary dict of the partition's label distribution.
    Useful for logging and for generating the 'data distribution vs compute' chart.

    Args:
        subset      : The Subset returned by dirichlet_partition().
        dataset     : The original full HuggingFace dataset.
        num_classes : Optional. If None, inferred from the shard itself.

    Returns:
        dict with keys: total_samples, label_counts, label_fractions, dominant_label
    """
    labels = np.array([dataset[int(i)]["labels"] for i in subset.indices])

    unique, counts = np.unique(labels, return_counts=True)
    label_counts   = dict(zip(unique.tolist(), counts.tolist()))
    label_fractions = {k: round(v / len(labels), 4) for k, v in label_counts.items()}
    dominant_label  = int(unique[np.argmax(counts)])

    return {
        "total_samples":    len(labels),
        "label_counts":     label_counts,
        "label_fractions":  label_fractions,
        "dominant_label":   dominant_label,
    }


def print_partition_report(device_tier: str, subset: Subset, dataset) -> None:
    """
    Prints a human-readable partition report for demo/judge visibility.
    """
    cfg  = TIER_PARTITION_CONFIG.get(device_tier, TIER_PARTITION_CONFIG[_DEFAULT_TIER])
    info = describe_partition(subset, dataset)

    print("\n" + "─" * 60)
    print(f"  DATA PARTITION REPORT  │  Tier: {device_tier}")
    print("─" * 60)
    print(f"  Profile     : {cfg['description']}")
    print(f"  α (Dirichlet): {cfg['alpha']}  │  Max fraction: {cfg['data_fraction'] * 100:.0f}% of corpus")
    print(f"  Shard size  : {info['total_samples']} samples")
    print(f"  Dominant label: #{info['dominant_label']}")
    print(f"  Label spread: {len(info['label_counts'])} unique classes in shard")

    # Simple ASCII bar chart of top-5 labels
    top5 = sorted(info["label_fractions"].items(), key=lambda x: -x[1])[:5]
    print("\n  Top-5 label distribution:")
    for label, frac in top5:
        bar = "█" * int(frac * 40)
        print(f"    Label {label:>3}: {bar} {frac * 100:.1f}%")
    print("─" * 60 + "\n")
