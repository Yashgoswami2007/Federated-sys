"""HTTP event sink for forwarding communication events to FastAPI.

This sink is deliberately best-effort. Communication and aggregation must keep
working even when the dashboard backend is offline, unauthenticated, or missing
database tables.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass
class HttpEventSink:
    """Maps communication-layer events to existing backend API routes."""

    base_url: str = "http://localhost:8000"
    token: str | None = None
    total_rounds: int = 1
    model_version: str = "local-mvp"
    model_name: str = "FusionNet Local MVP Adapter"
    timeout_s: float = 2.0
    verbose: bool = False
    failures: list[str] = field(default_factory=list)

    @classmethod
    def from_env(
        cls,
        total_rounds: int,
        base_url: str | None = None,
        token: str | None = None,
        verbose: bool = False,
    ) -> "HttpEventSink":
        return cls(
            base_url=base_url or os.getenv("BACKEND_URL", "http://localhost:8000"),
            token=token or os.getenv("HF_TOKEN"),
            total_rounds=total_rounds,
            verbose=verbose,
        )

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        try:
            if event_type == "round.started":
                self._handle_round_started(payload)
            elif event_type == "client.update_received":
                self._handle_client_update_received(payload)
            elif event_type == "round.aggregated":
                self._post_event("round.aggregated", "Coordinator completed FedAvg", payload)
            elif event_type == "global.update_published":
                self._handle_global_update_published(payload)
            elif event_type == "metrics.written":
                self._post_event("metrics.written", "Local MVP metrics written", payload)
            else:
                self._post_event(event_type, f"Communication event: {event_type}", payload)
        except Exception as exc:
            self._record_failure(f"{event_type}: unexpected sink error: {exc}")

    def _handle_round_started(self, payload: dict[str, Any]) -> None:
        round_num = int(payload["round"])
        expected_clients = int(payload["expected_clients"])
        self._request(
            "POST",
            "/api/rounds",
            {
                "round_number": round_num,
                "total_rounds": self.total_rounds,
                "expected_clients": expected_clients,
                "model_version": self.model_version,
            },
        )
        self._post_event(
            "round.started",
            f"Coordinator started round {round_num}",
            payload,
        )

    def _handle_client_update_received(self, payload: dict[str, Any]) -> None:
        round_num = int(payload["round"])
        received_clients = int(payload["received_clients"])
        expected_clients = int(payload["expected_clients"])
        progress = int((received_clients / expected_clients) * 90)
        client_id = payload["client_id"]
        hardware_tier = payload.get("hardware_tier", "Unknown")

        self._request(
            "POST",
            "/api/devices/register",
            {
                "client_id": client_id,
                "hardware_type": hardware_tier,
                "device_info": {
                    "source": "local_mvp",
                    "hardware_tier": hardware_tier,
                },
                "contribution_weight": 1.0,
                "region": "Local",
            },
        )

        self._request(
            "PATCH",
            f"/api/rounds/{round_num}",
            {
                "received_clients": received_clients,
                "progress": progress,
            },
        )

        metrics = payload.get("metrics", {})
        self._request(
            "POST",
            "/api/metrics",
            {
                "client_id": client_id,
                "round_number": round_num,
                "epoch": 1,
                "avg_loss": metrics.get("loss"),
                "accuracy": metrics.get("accuracy"),
                "epsilon_spent": metrics.get("epsilon"),
                "data_size": payload.get("num_samples"),
                "training_duration_s": metrics.get("train_time_s"),
                "partition_info": {
                    "hardware_tier": payload.get("hardware_tier"),
                    "source": "local_mvp",
                },
            },
        )

    def _handle_global_update_published(self, payload: dict[str, Any]) -> None:
        round_num = int(payload["round"])
        metrics = payload.get("metrics", {})
        artifact = payload.get("artifact", "")

        self._request(
            "PATCH",
            f"/api/rounds/{round_num}",
            {
                "status": "completed",
                "progress": 100,
                "global_model_path": artifact,
            },
        )

        self._request(
            "POST",
            "/api/metrics",
            {
                "client_id": "coordinator",
                "round_number": round_num,
                "epoch": 1,
                "avg_loss": metrics.get("avg_loss"),
                "accuracy": metrics.get("accuracy"),
                "epsilon_spent": metrics.get("epsilon_max"),
                "data_size": metrics.get("total_samples"),
                "partition_info": {
                    "clients": metrics.get("clients"),
                    "source": "local_mvp_global",
                },
            },
        )

        query = urlencode({"round_number": round_num, "hf_path": artifact})
        self._request(
            "POST",
            f"/api/models/global?{query}",
            {
                "name": self.model_name,
                "version": f"{self.model_version}.round-{round_num}",
                "accuracy": float(metrics.get("accuracy", 0.0)) * 100.0,
                "lastUpdated": "",
            },
        )

        self._post_event(
            "global.update_published",
            f"Global update published for round {round_num}",
            payload,
            severity="success",
        )

    def _post_event(
        self,
        event_type: str,
        message: str,
        payload: dict[str, Any],
        severity: str = "info",
    ) -> None:
        self._request(
            "POST",
            "/api/events",
            {
                "event_type": event_type,
                "severity": severity,
                "source": "fusionnet-comms",
                "message": message,
                "metadata_info": payload,
            },
        )

    def _request(self, method: str, path: str, payload: dict[str, Any]) -> bool:
        url = f"{self.base_url.rstrip('/')}{path}"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                response.read()
                return True
        except HTTPError as exc:
            self._record_failure(f"{method} {path} failed with HTTP {exc.code}")
        except URLError as exc:
            self._record_failure(f"{method} {path} failed: {exc.reason}")
        except OSError as exc:
            self._record_failure(f"{method} {path} failed: {exc}")
        return False

    def _record_failure(self, message: str) -> None:
        self.failures.append(message)
        if self.verbose:
            print(f"[backend-report] {message}")
