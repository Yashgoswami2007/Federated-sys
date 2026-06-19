import logging
import os
import sys
import tempfile
import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

# Allow import of auth from fusionnet-client root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from auth import get_token

logger = logging.getLogger(__name__)


class HFParameterServer:
    def __init__(self, repo_id: str, repo_type: str = "dataset"):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.api = HfApi(token=get_token())

    def upload_local_A_matrices(self, round_num: int, client_id: str, updates: list, data_size: int = 0):
        """Uploads local A matrices for a specific round to the HF repo.
        
        The upload includes dataset size metadata so the coordinator can perform
        properly weighted FedAvg (clients with more data get proportionally more
        influence on the global model).
        """
        payload = {"matrices": updates, "data_size": data_size}

        path_in_repo = f"round_{round_num}/{client_id}.pt"
        logger.info(f"Pushing local updates to HF Hub: {self.repo_id}/{path_in_repo}...")

        # Use a temp file for safe cleanup — avoids orphaned files if upload crashes
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            temp_path = tmp.name
            torch.save(payload, tmp)

        try:
            self.api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
            )
            logger.info("Upload successful.")
        finally:
            # Always clean up, even if upload fails
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def download_global_A_matrices(self, round_num: int):
        """
        Downloads the aggregated global A matrices for the given round.
        Returns None if not yet available.
        """
        path_in_repo = f"global/Global_A_round_{round_num}.pt"
        try:
            logger.info(f"Checking for global weights at {path_in_repo}...")
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=path_in_repo,
                repo_type=self.repo_type,
                local_dir="checkpoints/global",
                local_dir_use_symlinks=False,
            )
            logger.info("Successfully downloaded global A matrices.")
            return torch.load(local_path, weights_only=True)
        except EntryNotFoundError:
            logger.info("Global weights not found yet — coordinator may still be aggregating.")
            return None
        except Exception as e:
            logger.error(f"Error downloading global weights: {e}")
            return None
