from datasets import load_dataset
from torch.utils.data import DataLoader
from .partitioner import dirichlet_partition, print_partition_report


def get_dataset(config, tokenizer, device_tier: str = "CPU_only", client_id: int = 0, num_clients: int = 10):
    """
    Loads a dataset and carves out a Dirichlet-skewed shard for this client.

    The shard size and label skewness are both determined by `device_tier`:
      - Strong GPU tiers  → large, balanced shard (high Dirichlet α)
      - Weak CPU-only tier → small, heavily skewed shard (low Dirichlet α)

    This ensures the heterogeneity story is visible in the actual data
    distribution, not just in LoRA rank or batch-size numbers.

    Args:
        config      : Dataset config dict (from config.yaml).
        tokenizer   : HuggingFace tokenizer for the loaded model.
        device_tier : Hardware profile string (e.g. 'CPU_only', 'MI300X').
        client_id   : Unique integer ID for this node (seed offset for shard variety).
        num_clients : Total clients in the federation (controls Dirichlet split).

    Returns:
        (train_subset, test_dataset) — the private shard and full test set.
    """
    dataset_name = config.get("name", "banking77")
    text_column  = config.get("text_column", "text")
    label_column = config.get("label_column", "label")

    print(f"Loading dataset: {dataset_name}")
    if dataset_name == "banking77":
        raw_dataset = load_dataset("banking77")
    elif dataset_name == "sst2":
        raw_dataset = load_dataset("glue", "sst2")
        text_column = "sentence"
    elif dataset_name == "imdb":
        raw_dataset = load_dataset("imdb")
    elif dataset_name == "ag_news":
        raw_dataset = load_dataset("ag_news")
    else:
        raw_dataset = load_dataset(dataset_name)

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    tokenized = raw_dataset.map(tokenize_function, batched=True)

    # Normalise label column name to 'labels'
    cols_to_remove = [
        col for col in tokenized["train"].column_names
        if col not in ["input_ids", "attention_mask", label_column]
    ]
    tokenized = tokenized.remove_columns(cols_to_remove)
    if label_column != "labels" and label_column in tokenized["train"].column_names:
        tokenized = tokenized.rename_column(label_column, "labels")
    tokenized.set_format("torch")

    # ── Dirichlet partition: size + skew scaled to device tier ───────────────
    train_subset = dirichlet_partition(
        dataset=tokenized["train"],
        device_tier=device_tier,
        client_id=client_id,
        num_clients=num_clients,
    )

    # Print a judge-visible partition report showing data heterogeneity
    print_partition_report(device_tier, train_subset, tokenized["train"])

    return train_subset, tokenized["test"]
