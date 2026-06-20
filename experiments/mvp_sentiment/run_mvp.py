"""Local FusionNet MVP simulation.

This script proves the communication protocol before real networking is added:
multiple clients produce local weight updates, a coordinator receives them,
FedAvg aggregates the updates, and round metrics are saved for plotting.
"""

from __future__ import annotations

import argparse
import os
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

from fusionnet.comms import ClientUpdate, HttpEventSink, LocalCommunicationBackend


RESULTS_DIR = Path(__file__).resolve().parent / "results"
ADAPTER_KEY = "adapter.weight"


@dataclass(frozen=True)
class ClientProfile:
    client_id: str
    hardware_tier: str
    num_samples: int
    learning_rate: float
    noise_scale: float


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

    def __init__(self, clients: list[LocalClient], results_dir: Path, event_sink: Any = None):
        self.clients = clients
        self.results_dir = results_dir
        self.comms = LocalCommunicationBackend(
            results_dir=results_dir,
            expected_clients=len(clients),
            event_sink=event_sink,
        )
        self.metrics: list[dict[str, Any]] = []

    def prepare_results_dir(self) -> None:
        self.comms.prepare()

    def run(self, rounds: int, adapter_shape: tuple[int, int]) -> dict[str, Any]:
        self.prepare_results_dir()
        global_weights = {ADAPTER_KEY: zeros(adapter_shape)}

        print("FusionNet MVP Demo")
        print(f"Clients: {len(self.clients)} | Rounds: {rounds}")
        print("-" * 56)

        for round_num in range(1, rounds + 1):
            self.comms.start_round(round_num, global_weights)
            updates = self.collect_client_updates(global_weights, round_num)
            global_weights = self.comms.aggregate(round_num)
            print(f"Round {round_num}: aggregated global weights with FedAvg")
            round_metrics = self.summarize_round(round_num, updates, global_weights)
            self.metrics.append(round_metrics)

            self.comms.publish_global_update(round_num, global_weights, round_metrics)
            self.print_round_summary(round_metrics)

        metrics_path = self.comms.write_metrics(self.metrics)
        print(f"Saved metrics to {metrics_path.relative_to(REPO_ROOT)}")
        return global_weights

    def collect_client_updates(
        self,
        global_weights: dict[str, Any],
        round_num: int,
    ) -> list[ClientUpdate]:
        updates = []
        for client in self.clients:
            update = client.train(global_weights, round_num)
            self.comms.submit_update(update)
            updates.append(update)
        print(f"Round {round_num}: received {len(updates)} client updates")
        return updates

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

    def print_round_summary(self, metrics: dict[str, Any]) -> None:
        print(
            "Round {round}: loss {avg_loss:.4f}, accuracy {accuracy:.2%}, "
            "epsilon {epsilon_max:.2f}, samples {total_samples}".format(**metrics)
        )
        print("-" * 56)


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


def mean_scalar(value: Any) -> float:
    if torch is not None:
        return float(value.mean().item())
    return float(np.mean(value))


def vector_norm(value: Any) -> float:
    if torch is not None:
        return float(torch.linalg.vector_norm(value).item())
    return float(np.linalg.norm(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local FusionNet MVP simulation")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated rounds")
    parser.add_argument("--adapter-rows", type=int, default=8, help="Simulated adapter tensor rows")
    parser.add_argument("--adapter-cols", type=int, default=8, help="Simulated adapter tensor columns")
    parser.add_argument(
        "--report-backend",
        action="store_true",
        help="Forward communication events to the FastAPI backend",
    )
    parser.add_argument(
        "--backend-url",
        default=os.getenv("BACKEND_URL", "http://localhost:8000"),
        help="FastAPI backend URL for --report-backend",
    )
    parser.add_argument(
        "--backend-token",
        default=os.getenv("HF_TOKEN"),
        help="Bearer token for backend auth; defaults to HF_TOKEN",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1")
    if args.adapter_rows < 1 or args.adapter_cols < 1:
        raise ValueError("--adapter-rows and --adapter-cols must be positive")

    adapter_shape = (args.adapter_rows, args.adapter_cols)
    clients = build_clients(adapter_shape)
    event_sink = None
    if args.report_backend:
        event_sink = HttpEventSink.from_env(
            total_rounds=args.rounds,
            base_url=args.backend_url,
            token=args.backend_token,
            verbose=True,
        )
        print(f"Backend reporting enabled: {args.backend_url}")

    coordinator = LocalCoordinator(clients, RESULTS_DIR, event_sink=event_sink)
    final_weights = coordinator.run(args.rounds, adapter_shape)

    final_path = RESULTS_DIR / f"global_round_{args.rounds}.pt"
    final_norm = vector_norm(final_weights[ADAPTER_KEY])
    print(f"Saved final global weights to {final_path.relative_to(REPO_ROOT)}")
    print(f"Final global update norm: {final_norm:.4f}")


if __name__ == "__main__":
    main()
