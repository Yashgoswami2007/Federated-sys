import time
import torch
import sys
import os
from huggingface_hub import HfApi, hf_hub_download

# Ensure fusionnet is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fusionnet.core.aggregator import fed_avg

class HFCoordinator:
    def __init__(self, repo_id: str, num_clients: int, repo_type: str = "dataset"):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.num_clients = num_clients
        self.api = HfApi()
        
    def aggregate_round(self, round_num: int):
        print(f"\n=== Starting Coordinator for Round {round_num} ===")
        print(f"Watching Hugging Face Repo: {self.repo_id}")
        print(f"Waiting for {self.num_clients} client updates to appear in 'round_{round_num}/'...")
        
        while True:
            try:
                files = self.api.list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type)
                round_files = [f for f in files if f.startswith(f"round_{round_num}/") and f.endswith(".pt")]
                
                print(f"  [Status] Found {len(round_files)}/{self.num_clients} client updates.", end='\r')
                
                if len(round_files) >= self.num_clients:
                    print(f"\nFound {len(round_files)} updates! Proceeding with aggregation.")
                    break
            except Exception as e:
                print(f"\nError querying repo: {e}. Retrying...")
                
            time.sleep(10)
            
        # Download updates
        client_updates = []
        for file in round_files:
            print(f"Downloading {file}...")
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=file,
                repo_type=self.repo_type,
                local_dir="checkpoints/coordinator_tmp",
                local_dir_use_symlinks=False
            )
            client_updates.append(torch.load(local_path))
            
        # Aggregate (FedAvg)
        print("Aggregating A matrices via FedAvg...")
        num_layers = len(client_updates[0])
        global_tensors = []
        
        for layer_idx in range(num_layers):
            layer_tensors = []
            for client_idx in range(len(client_updates)):
                layer_tensors.append({"a_matrix": client_updates[client_idx][layer_idx]})
                
            # Treat them equally for this MVP. In production, we'd weight by dataset sizes.
            sizes = [1] * len(layer_tensors)
            avg_dict = fed_avg(layer_tensors, sizes)
            global_tensors.append(avg_dict["a_matrix"])
            
        # Upload Global A
        global_path = f"global/Global_A_round_{round_num}.pt"
        temp_file = f"temp_Global_A_round_{round_num}.pt"
        torch.save(global_tensors, temp_file)
        
        print(f"Uploading aggregated Global weights to {global_path}...")
        self.api.upload_file(
            path_or_fileobj=temp_file,
            path_in_repo=global_path,
            repo_id=self.repo_id,
            repo_type=self.repo_type
        )
        os.remove(temp_file)
        print(f"Round {round_num} aggregation complete! Global weights are live.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hugging Face Serverless FL Coordinator")
    parser.add_argument("--repo-id", type=str, required=True, help="HF Repo ID (e.g. Yashgoswami2007/fusionnet)")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of clients to wait for")
    parser.add_argument("--rounds", type=int, default=1, help="Total federated rounds to run")
    args = parser.parse_args()
    
    coordinator = HFCoordinator(args.repo_id, args.num_clients)
    for r in range(1, args.rounds + 1):
        coordinator.aggregate_round(r)
