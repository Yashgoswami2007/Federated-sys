import time
import torch
import sys
import os
import httpx
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

# Load HF token from .env in repo root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Add it to your .env file.")

# Ensure fusionnet core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fusionnet.core.aggregator import fed_avg


class HFCoordinator:
    def __init__(self, repo_id: str, num_clients: int, repo_type: str = "dataset"):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.num_clients = num_clients
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
            print(f"Backend report failed ({path}): {e}")

    def aggregate_round(self, round_num: int):
        print(f"\n=== Coordinator: Round {round_num} ===")
        print(f"Repo: {self.repo_id}")
        print(f"Waiting for {self.num_clients} client updates in 'round_{round_num}/'...")
        
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

        # Poll until all clients have uploaded
        while True:
            try:
                files = self.api.list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type)
                round_files = [
                    f for f in files
                    if f.startswith(f"round_{round_num}/") and f.endswith(".pt")
                ]
                print(f"  [Status] {len(round_files)}/{self.num_clients} updates received.", end="\r")

                if len(round_files) >= self.num_clients:
                    print(f"\nAll {len(round_files)} updates received. Aggregating...")
                    break
                else:
                    self._report_backend("PATCH", f"/api/rounds/{round_num}", {
                        "received_clients": len(round_files),
                        "progress": int((len(round_files) / self.num_clients) * 100)
                    })
            except Exception as e:
                print(f"\nError querying repo: {e}. Retrying in 10s...")

            time.sleep(10)

        # Download all client updates
        client_updates = []
        for file in round_files:
            print(f"Downloading {file}...")
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=file,
                repo_type=self.repo_type,
                local_dir="checkpoints/coordinator_tmp",
                local_dir_use_symlinks=False,
            )
            client_updates.append(torch.load(local_path, weights_only=True))

        # FedAvg across layers
        print("Running FedAvg on A matrices...")
        num_layers = len(client_updates[0])
        global_tensors = []

        for layer_idx in range(num_layers):
            layer_tensors = [
                {"a_matrix": client_updates[c][layer_idx]}
                for c in range(len(client_updates))
            ]
            # Equal weighting for MVP — production would weight by dataset size
            sizes = [1] * len(layer_tensors)
            avg_dict = fed_avg(layer_tensors, sizes)
            global_tensors.append(avg_dict["a_matrix"])

        # Upload aggregated global A
        global_path = f"global/Global_A_round_{round_num}.pt"
        temp_file = f"temp_Global_A_round_{round_num}.pt"
        torch.save(global_tensors, temp_file)

        print(f"Uploading global weights to {global_path}...")
        self.api.upload_file(
            path_or_fileobj=temp_file,
            path_in_repo=global_path,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
        )
        os.remove(temp_file)
        print(f"Round {round_num} complete. Global weights live at {self.repo_id}/{global_path}")
        
        # Report completion
        self._report_backend("PATCH", f"/api/rounds/{round_num}", {
            "status": "completed",
            "progress": 100,
            "global_model_path": f"{self.repo_id}/{global_path}"
        })
        self._report_backend("POST", "/api/models/global", {
            "name": "TinyLlama-1.1B-Chat-AFLoRA",
            "version": f"v0.7.{round_num}",
            "accuracy": 94.2 + (round_num * 0.1), # Mock accuracy increase
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
    parser.add_argument("--repo-id",     type=str, default="yash-goswami/fusionnet-coordinator")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--rounds",      type=int, default=1)
    args = parser.parse_args()

    coordinator = HFCoordinator(args.repo_id, args.num_clients)
    for r in range(1, args.rounds + 1):
        coordinator.aggregate_round(r)
