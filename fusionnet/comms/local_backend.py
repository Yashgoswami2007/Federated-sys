"""Local communication backend for FusionNet MVP simulations.

The local backend behaves like a tiny in-process parameter server:
clients submit updates, the coordinator waits for the expected number of
updates, and then publishes a global update for the round.

This is intentionally transport-shaped so HTTP, HF Hub, and RCCL backends can
reuse the same ClientUpdate/GlobalUpdate message contract later.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

class RoundTimeoutError(RuntimeError):
    pass

if torch is not None:
    from fusionnet.core.aggregator import fed_avg as torch_fed_avg
else:
    torch_fed_avg = None


class BackendEventSink(Protocol):
    """Optional sidecar for telemetry backends."""

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Receive a protocol event without affecting training."""


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


@dataclass
class GlobalUpdate:
    round_num: int
    weights: dict[str, Any]
    metrics: dict[str, Any]

    def to_artifact(self) -> dict[str, Any]:
        return {
            "round": self.round_num,
            "weights": {key: to_cpu(value) for key, value in self.weights.items()},
            "metrics": self.metrics,
        }


@dataclass
class LocalCommunicationBackend:
    """Stores local round updates and artifacts for an MVP coordinator."""

    results_dir: Path
    expected_clients: int
    event_sink: BackendEventSink | None = None
    updates_by_round: dict[int, list[ClientUpdate]] = field(default_factory=dict)
    global_updates: dict[int, GlobalUpdate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.results_dir = Path(self.results_dir)
        self.client_updates_dir = self.results_dir / "client_updates"

    def prepare(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.client_updates_dir.mkdir(parents=True, exist_ok=True)
        
        # Recover state from disk if exists
        for file_path in self.client_updates_dir.glob("round_*.pt"):
            try:
                data = load_artifact(file_path)
                update = ClientUpdate(
                    client_id=data["client_id"],
                    round_num=data["round"],
                    num_samples=data["num_samples"],
                    hardware_tier=data["hardware_tier"],
                    weights=data["weights"],
                    metrics=data["metrics"],
                )
                updates = self.updates_by_round.setdefault(update.round_num, [])
                if not any(existing.client_id == update.client_id for existing in updates):
                    updates.append(update)
            except Exception as e:
                print(f"Warning: Failed to load artifact {file_path}: {e}")

    def start_round(self, round_num: int, global_weights: dict[str, Any]) -> None:
        self.prepare()
        self.updates_by_round[round_num] = []
        self._emit(
            "round.started",
            {
                "round": round_num,
                "expected_clients": self.expected_clients,
                "global_weight_keys": list(global_weights.keys()),
            },
        )

    def submit_update(self, update: ClientUpdate) -> None:
        updates = self.updates_by_round.setdefault(update.round_num, [])
        if any(existing.client_id == update.client_id for existing in updates):
            raise ValueError(
                f"Client {update.client_id} already submitted round {update.round_num}"
            )

        updates.append(update)
        self._save_client_update(update)
        self._emit(
            "client.update_received",
            {
                "round": update.round_num,
                "client_id": update.client_id,
                "received_clients": len(updates),
                "expected_clients": self.expected_clients,
                "num_samples": update.num_samples,
                "hardware_tier": update.hardware_tier,
                "metrics": update.metrics,
            },
        )

    def wait_for_updates(self, round_num: int, timeout_sec: float = 60.0, min_clients: int | None = None) -> list[ClientUpdate]:
        if min_clients is None:
            min_clients = self.expected_clients
            
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            updates = self.updates_by_round.get(round_num, [])
            if len(updates) >= self.expected_clients:
                return list(updates)
            time.sleep(1.0)
            
        updates = self.updates_by_round.get(round_num, [])
        if len(updates) >= min_clients:
            print(f"Warning: Timeout reached. Proceeding with partial aggregation ({len(updates)}/{self.expected_clients} clients).")
            return list(updates)
            
        raise RoundTimeoutError(
            f"Round {round_num} failed: Expected at least {min_clients} updates, "
            f"but received {len(updates)} before timeout of {timeout_sec}s."
        )

    def aggregate(self, round_num: int, timeout_sec: float = 60.0, min_clients: int | None = None) -> dict[str, Any]:
        updates = self.wait_for_updates(round_num, timeout_sec=timeout_sec, min_clients=min_clients)
        client_weights = [update.weights for update in updates]
        client_sizes = [update.num_samples for update in updates]
        global_weights = aggregate_weights(client_weights, client_sizes)
        if global_weights is None:
            raise RuntimeError(f"Round {round_num} aggregation failed: no updates")

        self._emit(
            "round.aggregated",
            {
                "round": round_num,
                "clients": len(updates),
                "total_samples": sum(update.num_samples for update in updates),
            },
        )
        return global_weights

    def publish_global_update(
        self,
        round_num: int,
        weights: dict[str, Any],
        metrics: dict[str, Any],
    ) -> GlobalUpdate:
        global_update = GlobalUpdate(round_num=round_num, weights=weights, metrics=metrics)
        self.global_updates[round_num] = global_update
        save_artifact(weights, self.results_dir / f"global_round_{round_num}.pt")
        self._emit(
            "global.update_published",
            {
                "round": round_num,
                "artifact": str(self.results_dir / f"global_round_{round_num}.pt"),
                "metrics": metrics,
            },
        )
        return global_update

    def write_metrics(self, metrics: list[dict[str, Any]]) -> Path:
        metrics_path = self.results_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self._emit("metrics.written", {"path": str(metrics_path), "rounds": len(metrics)})
        return metrics_path

    def _save_client_update(self, update: ClientUpdate) -> None:
        filename = f"round_{update.round_num}_{update.client_id}.pt"
        save_artifact(update.to_artifact(), self.client_updates_dir / filename)

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink.emit(event_type, payload)


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


def to_cpu(value: Any) -> Any:
    if torch is not None:
        return value.cpu()
    return np.array(value, copy=True)


def save_artifact(payload: Any, path: Path) -> None:
    if torch is not None:
        torch.save(payload, path)
        return

    with path.open("wb") as artifact_file:
        pickle.dump(payload, artifact_file)


def load_artifact(path: Path) -> Any:
    if torch is not None:
        return torch.load(path, weights_only=False)
    with path.open("rb") as artifact_file:
        return pickle.load(artifact_file)
