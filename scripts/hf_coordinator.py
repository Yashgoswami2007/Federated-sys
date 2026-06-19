import logging
import tempfile
import time
import torch
import sys
import os
import httpx
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hf_coordinator")

# Load HF token from .env in repo root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Add it to your .env file.")

# Ensure fusionnet core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fusionnet.core.aggregator import fed_avg


class HFCoordinator:
    def __init__(self, repo_id: str, num_clients: int, repo_type: str = "dataset",
                 timeout_seconds: int = 1800, min_clients: int = None):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.num_clients = num_clients
        self.timeout_seconds = timeout_seconds
        # Default min_clients to num_clients (wait for all), but allow partial aggregation
        self.min_clients = min_clients if min_clients is not None else num_clients
        self.api = HfApi(token=HF_TOKEN)
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        self.backend_enabled = os.getenv("BACKEND_ENABLED", "True").lower() in ["true", "1", "yes"]

    def _report_backend(self, method: str, path: str, json_data: dict):
        if not self.backend_enabled:
            return
        try:
            url = f"{self.backend_url}{path}"
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            if method == "POST":
                httpx.post(url, json=json_data, headers=headers, timeout=5.0)
            elif method == "PATCH":
                httpx.patch(url, json=json_data, headers=headers, timeout=5.0)
        except Exception as e:
            logger.debug(f"Backend report failed ({path}): {e}")

    def _parse_client_payload(self, raw_payload):
        """Parse a client upload, handling both old (list) and new (dict) formats.
        
        Old format: list of tensors (A matrices only, no metadata)
        New format: {"matrices": [...], "data_size": int}
        
        Returns:
            (matrices_list, data_size)
        """
        if isinstance(raw_payload, dict):
            return raw_payload["matrices"], raw_payload.get("data_size", 1)
        else:
            # Legacy format: plain list of tensors, assume data_size=1 (equal weighting)
            logger.warning("Client uploaded in legacy format (no data_size metadata). Using equal weight.")
            return raw_payload, 1

    def aggregate_round(self, round_num: int):
        logger.info(f"=== Coordinator: Round {round_num} ===")
        logger.info(f"Repo: {self.repo_id}")
        logger.info(f"Waiting for {self.num_clients} clients (min: {self.min_clients}) "
                     f"in 'round_{round_num}/' (timeout: {self.timeout_seconds}s)...")
        
        # Report round start
        self._report_backend("POST", "/api/rounds", {
            "round_number": round_num,
            "total_rounds": 10,
            "expected_clients": self.num_clients,
            "model_version": "v0.7.0"
        })
        self._report_backend("POST", "/api/events", {
            "event_type": "round.started",
            "message": f"Coordinator started round {round_num}",
            "severity": "info",
            "metadata_info": {"expected_clients": self.num_clients}
        })

        # Poll until all clients have uploaded OR timeout is reached
        start_time = time.time()
        round_files = []

        while True:
            elapsed = time.time() - start_time

            try:
                files = self.api.list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type)
                round_files = [
                    f for f in files
                    if f.startswith(f"round_{round_num}/") and f.endswith(".pt")
                ]
                logger.info(f"  [Status] {len(round_files)}/{self.num_clients} updates received "
                            f"({elapsed:.0f}s elapsed)")

                if len(round_files) >= self.num_clients:
                    logger.info(f"All {len(round_files)} updates received. Aggregating...")
                    break

                self._report_backend("PATCH", f"/api/rounds/{round_num}", {
                    "received_clients": len(round_files),
                    "progress": int((len(round_files) / self.num_clients) * 100)
                })

            except Exception as e:
                logger.error(f"Error querying repo: {e}. Retrying in 10s...")

            # Check timeout
            if elapsed >= self.timeout_seconds:
                if len(round_files) >= self.min_clients:
                    logger.warning(
                        f"Timeout reached ({self.timeout_seconds}s). "
                        f"Proceeding with {len(round_files)}/{self.num_clients} clients "
                        f"(meets min_clients={self.min_clients})."
                    )
                    self._report_backend("POST", "/api/events", {
                        "event_type": "round.timeout",
                        "message": f"Round {round_num} timed out with {len(round_files)}/{self.num_clients} clients",
                        "severity": "warning",
                        "metadata_info": {"received": len(round_files), "expected": self.num_clients}
                    })
                    break
                else:
                    logger.error(
                        f"Timeout reached ({self.timeout_seconds}s) with only "
                        f"{len(round_files)}/{self.min_clients} minimum clients. "
                        f"Skipping round {round_num}."
                    )
                    self._report_backend("POST", "/api/events", {
                        "event_type": "round.failed",
                        "message": f"Round {round_num} failed: insufficient clients ({len(round_files)}/{self.min_clients})",
                        "severity": "error"
                    })
                    return  # Skip this round

            time.sleep(10)

        if not round_files:
            logger.error(f"No client updates found for round {round_num}. Skipping.")
            return

        # Download all client updates
        client_matrices = []
        client_data_sizes = []

        for file in round_files:
            logger.info(f"Downloading {file}...")
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=file,
                repo_type=self.repo_type,
                local_dir="checkpoints/coordinator_tmp",
                local_dir_use_symlinks=False,
            )
            raw_payload = torch.load(local_path, weights_only=True)
            matrices, data_size = self._parse_client_payload(raw_payload)
            client_matrices.append(matrices)
            client_data_sizes.append(data_size)

        # Data-size weighted FedAvg across layers
        logger.info(f"Running weighted FedAvg on A matrices "
                     f"(data sizes: {client_data_sizes})...")
        num_layers = len(client_matrices[0])
        global_tensors = []

        for layer_idx in range(num_layers):
            layer_tensors = [
                {"a_matrix": client_matrices[c][layer_idx]}
                for c in range(len(client_matrices))
            ]
            # Use actual data sizes for weighted averaging
            avg_dict = fed_avg(layer_tensors, client_data_sizes)
            global_tensors.append(avg_dict["a_matrix"])

        # Upload aggregated global A using a temp file for safe cleanup
        global_path = f"global/Global_A_round_{round_num}.pt"

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            temp_file = tmp.name
            torch.save(global_tensors, tmp)

        try:
            logger.info(f"Uploading global weights to {global_path}...")
            self.api.upload_file(
                path_or_fileobj=temp_file,
                path_in_repo=global_path,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
            )
        finally:
            os.remove(temp_file)

        logger.info(f"Round {round_num} complete. Global weights live at {self.repo_id}/{global_path}")
        
        # Report completion
        self._report_backend("PATCH", f"/api/rounds/{round_num}", {
            "status": "completed",
            "progress": 100,
            "global_model_path": f"{self.repo_id}/{global_path}"
        })
        self._report_backend("POST", "/api/models/global", {
            "name": "TinyLlama-1.1B-Chat-AFLoRA",
            "version": f"v0.7.{round_num}",
            "accuracy": 94.2 + (round_num * 0.1),  # Mock accuracy increase
            "round_number": round_num,
            "hf_path": f"{self.repo_id}/{global_path}"
        })
        self._report_backend("POST", "/api/events", {
            "event_type": "model.global_updated",
            "message": f"Global model updated for round {round_num}",
            "severity": "success",
            "metadata_info": {"path": f"{self.repo_id}/{global_path}"}
        })
        self._report_backend("POST", "/api/events", {
            "event_type": "round.completed",
            "message": f"Round {round_num} completed successfully",
            "severity": "success"
        })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FusionNet HF Serverless Coordinator")
    parser.add_argument("--repo-id",      type=str, default="yash-goswami/fusionnet-coordinator")
    parser.add_argument("--num-clients",  type=int, default=3)
    parser.add_argument("--min-clients",  type=int, default=None,
                        help="Minimum clients to proceed after timeout (default: same as --num-clients)")
    parser.add_argument("--rounds",       type=int, default=1)
    parser.add_argument("--timeout",      type=int, default=1800,
                        help="Max seconds to wait for client uploads per round (default: 1800 = 30 min)")
    args = parser.parse_args()

    coordinator = HFCoordinator(
        args.repo_id, args.num_clients,
        timeout_seconds=args.timeout,
        min_clients=args.min_clients,
    )
    for r in range(1, args.rounds + 1):
        coordinator.aggregate_round(r)
