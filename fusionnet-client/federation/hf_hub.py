import os
import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

class HFParameterServer:
    def __init__(self, repo_id: str, repo_type: str = "dataset"):
        """
        Initializes connection to the Hugging Face Serverless Parameter Server.
        Requires huggingface-cli login.
        """
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.api = HfApi()

    def upload_local_A_matrices(self, round_num: int, client_id: str, updates: list):
        """
        Uploads local A matrices for a specific round to the HF repo.
        Each update in `updates` is a tensor representing the A matrix of a layer.
        """
        temp_file = f"temp_{client_id}_round_{round_num}.pt"
        torch.save(updates, temp_file)
        
        path_in_repo = f"round_{round_num}/{client_id}.pt"
        print(f"Pushing local updates to HF Hub: {self.repo_id}/{path_in_repo}...")
        
        self.api.upload_file(
            path_or_fileobj=temp_file,
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type=self.repo_type
        )
        os.remove(temp_file)
        print("Upload successful.")
        
    def download_global_A_matrices(self, round_num: int):
        """
        Downloads the aggregated Global A matrices for the given round.
        Returns None if the file is not yet available on the Hub.
        """
        path_in_repo = f"global/Global_A_round_{round_num}.pt"
        try:
            print(f"Checking for Global weights at {path_in_repo}...")
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=path_in_repo,
                repo_type=self.repo_type,
                local_dir="checkpoints/global",
                local_dir_use_symlinks=False
            )
            print("Successfully downloaded Global A matrices.")
            return torch.load(local_path)
        except EntryNotFoundError:
            print("Global weights not found yet. The coordinator might still be aggregating.")
            return None
        except Exception as e:
            print(f"Error checking global weights: {e}")
            return None
