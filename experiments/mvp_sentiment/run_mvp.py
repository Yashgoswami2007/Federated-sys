"""Local FusionNet MVP simulation.

This script proves the communication protocol before real networking is added:
multiple clients produce local weight updates, a coordinator receives them,
FedAvg aggregates the updates, and round metrics are saved for plotting.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if torch is not None:
    from fusionnet.core.aggregator import fed_avg as torch_fed_avg
else:
    torch_fed_avg = None


RESULTS_DIR = Path(__file__).resolve().parent / "results"
ADAPTER_KEY = "adapter.weight"


@dataclass(frozen=True)
class ClientProfile:
    client_id: str
    hardware_tier: str
    num_samples: int
    learning_rate: float
    noise_scale: float


@dataclass
class ClientUpdate:
    client_id: str
    round_num: int
    num_samples: int
    hardware_tier: str
    weights: dict[str, Any]
    metrics: dict[str, float]

    def to_artifact(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "round": self.round_num,
            "num_samples": self.num_samples,
            "hardware_tier": self.hardware_tier,
            "weights": {key: to_cpu(value) for key, value in self.weights.items()},
            "metrics": self.metrics,
        }


class LocalClient:
    """Simulates one edge node training on private local data."""

    def __init__(self, profile: ClientProfile, target_weights: Any):
        self.profile = profile
        self.target_weights = target_weights

    def train(self, global_weights: dict[str, Any], round_num: int) -> ClientUpdate:
        start = time.perf_counter()
        current = global_weights[ADAPTER_KEY]

        noise = randn_like(current, seed=1000 + round_num * 97 + self.profile.num_samples)
        noise = noise * (self.profile.noise_scale / round_num)

        local_delta = (self.target_weights - current) * self.profile.learning_rate
        updated = current + local_delta + noise

        loss = mean_scalar((updated - self.target_weights) ** 2)
        accuracy = max(0.0, min(0.99, 1.0 - loss))
        epsilon = round(0.18 * round_num + self.profile.noise_scale, 4)
        train_time_s = time.perf_counter() - start

        return ClientUpdate(
            client_id=self.profile.client_id,
            round_num=round_num,
            num_samples=self.profile.num_samples,
            hardware_tier=self.profile.hardware_tier,
            weights={ADAPTER_KEY: clone_array(updated)},
            metrics={
                "loss": round(loss, 6),
                "accuracy": round(accuracy, 6),
                "epsilon": epsilon,
                "train_time_s": round(train_time_s, 6),
            },
        )


class LocalCoordinator:
    """Coordinates rounds for the local network simulation."""

    def __init__(self, clients: list[LocalClient], results_dir: Path):
        self.clients = clients
        self.results_dir = results_dir
        self.client_updates_dir = results_dir / "client_updates"
        self.metrics: list[dict[str, Any]] = []

    def prepare_results_dir(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.client_updates_dir.mkdir(parents=True, exist_ok=True)

    def run(self, rounds: int, adapter_shape: tuple[int, int]) -> dict[str, Any]:
        self.prepare_results_dir()
        global_weights = {ADAPTER_KEY: zeros(adapter_shape)}

        print("FusionNet MVP Demo")
        print(f"Clients: {len(self.clients)} | Rounds: {rounds}")
        print("-" * 56)

        for round_num in range(1, rounds + 1):
            updates = self.collect_client_updates(global_weights, round_num)
            global_weights = self.aggregate(round_num, updates)
            round_metrics = self.summarize_round(round_num, updates, global_weights)
            self.metrics.append(round_metrics)

            self.save_round_artifacts(round_num, updates, global_weights)
            self.print_round_summary(round_metrics)

        self.write_metrics()
        return global_weights

    def collect_client_updates(
        self,
        global_weights: dict[str, Any],
        round_num: int,
    ) -> list[ClientUpdate]:
        updates = [client.train(global_weights, round_num) for client in self.clients]
        print(f"Round {round_num}: received {len(updates)} client updates")
        return updates

    def aggregate(
        self,
        round_num: int,
        updates: list[ClientUpdate],
    ) -> dict[str, Any]:
        client_weights = [update.weights for update in updates]
        client_sizes = [update.num_samples for update in updates]
        global_weights = aggregate_weights(client_weights, client_sizes)
        if global_weights is None:
            raise RuntimeError(f"Round {round_num} aggregation failed: no client updates")

        print(f"Round {round_num}: aggregated global weights with FedAvg")
        return global_weights

    def summarize_round(
        self,
        round_num: int,
        updates: list[ClientUpdate],
        global_weights: dict[str, Any],
    ) -> dict[str, Any]:
        total_samples = sum(update.num_samples for update in updates)
        avg_loss = sum(update.metrics["loss"] * update.num_samples for update in updates) / total_samples
        avg_accuracy = max(0.0, min(0.99, 1.0 - avg_loss))
        epsilon_max = max(update.metrics["epsilon"] for update in updates)
        update_norm = vector_norm(global_weights[ADAPTER_KEY])

        return {
            "round": round_num,
            "avg_loss": round(avg_loss, 6),
            "accuracy": round(avg_accuracy, 6),
            "clients": len(updates),
            "total_samples": total_samples,
            "epsilon_max": round(epsilon_max, 6),
            "global_update_norm": round(update_norm, 6),
            "client_metrics": [
                {
                    "client_id": update.client_id,
                    "hardware_tier": update.hardware_tier,
                    "num_samples": update.num_samples,
                    **update.metrics,
                }
                for update in updates
            ],
        }

    def save_round_artifacts(
        self,
        round_num: int,
        updates: list[ClientUpdate],
        global_weights: dict[str, Any],
    ) -> None:
        save_artifact(global_weights, self.results_dir / f"global_round_{round_num}.pt")

        for update in updates:
            filename = f"round_{round_num}_{update.client_id}.pt"
            save_artifact(update.to_artifact(), self.client_updates_dir / filename)

    def print_round_summary(self, metrics: dict[str, Any]) -> None:
        print(
            "Round {round}: loss {avg_loss:.4f}, accuracy {accuracy:.2%}, "
            "epsilon {epsilon_max:.2f}, samples {total_samples}".format(**metrics)
        )
        print("-" * 56)

    def write_metrics(self) -> None:
        metrics_path = self.results_dir / "metrics.json"
        metrics_path.write_text(json.dumps(self.metrics, indent=2), encoding="utf-8")
        print(f"Saved metrics to {metrics_path.relative_to(REPO_ROOT)}")


def build_clients(adapter_shape: tuple[int, int]) -> list[LocalClient]:
    profiles = [
        ClientProfile("client_0", "CPU_only", 400, 0.35, 0.035),
        ClientProfile("client_1", "Steam_Deck", 900, 0.45, 0.025),
        ClientProfile("client_2", "RX_7900_XTX", 1600, 0.55, 0.015),
    ]

    clients: list[LocalClient] = []
    for idx, profile in enumerate(profiles):
        target = randn(adapter_shape, seed=42 + idx)
        target = target * (0.25 + idx * 0.05)
        clients.append(LocalClient(profile, target))
    return clients


def zeros(shape: tuple[int, int]) -> Any:
    if torch is not None:
        return torch.zeros(shape, dtype=torch.float32)
    return np.zeros(shape, dtype=np.float32)


def randn(shape: tuple[int, int], seed: int) -> Any:
    if torch is not None:
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(shape, generator=generator, dtype=torch.float32)
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def randn_like(value: Any, seed: int) -> Any:
    if torch is not None:
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(
            value.shape,
            generator=generator,
            dtype=value.dtype,
            device=value.device,
        )
    return np.random.default_rng(seed).standard_normal(value.shape).astype(value.dtype)


def clone_array(value: Any) -> Any:
    if torch is not None:
        return value.detach().clone()
    return np.array(value, copy=True)


def to_cpu(value: Any) -> Any:
    if torch is not None:
        return value.cpu()
    return np.array(value, copy=True)


def mean_scalar(value: Any) -> float:
    if torch is not None:
        return float(value.mean().item())
    return float(np.mean(value))


def vector_norm(value: Any) -> float:
    if torch is not None:
        return float(torch.linalg.vector_norm(value).item())
    return float(np.linalg.norm(value))


def aggregate_weights(client_weights: list[dict[str, Any]], client_sizes: list[int]) -> dict[str, Any] | None:
    if torch_fed_avg is not None:
        return torch_fed_avg(client_weights, client_sizes)

    if not client_weights or not client_sizes:
        return None

    if len(client_weights) != len(client_sizes):
        raise ValueError("Number of weight updates must match number of data sizes.")

    total_samples = float(sum(client_sizes))
    averaged: dict[str, Any] = {}
    for key in client_weights[0]:
        weighted_sum = np.zeros_like(client_weights[0][key], dtype=np.float32)
        for weights, sample_count in zip(client_weights, client_sizes):
            weighted_sum += weights[key].astype(np.float32) * (sample_count / total_samples)
        averaged[key] = weighted_sum
    return averaged


def save_artifact(payload: Any, path: Path) -> None:
    if torch is not None:
        torch.save(payload, path)
        return

    with path.open("wb") as artifact_file:
        pickle.dump(payload, artifact_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local FusionNet MVP simulation")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated rounds")
    parser.add_argument("--adapter-rows", type=int, default=8, help="Simulated adapter tensor rows")
    parser.add_argument("--adapter-cols", type=int, default=8, help="Simulated adapter tensor columns")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1")
    if args.adapter_rows < 1 or args.adapter_cols < 1:
        raise ValueError("--adapter-rows and --adapter-cols must be positive")

    adapter_shape = (args.adapter_rows, args.adapter_cols)
    clients = build_clients(adapter_shape)
    coordinator = LocalCoordinator(clients, RESULTS_DIR)
    final_weights = coordinator.run(args.rounds, adapter_shape)

    final_path = RESULTS_DIR / f"global_round_{args.rounds}.pt"
    final_norm = vector_norm(final_weights[ADAPTER_KEY])
    print(f"Saved final global weights to {final_path.relative_to(REPO_ROOT)}")
    print(f"Final global update norm: {final_norm:.4f}")


if __name__ == "__main__":
    main()
