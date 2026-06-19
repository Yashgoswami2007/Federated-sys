import os
import torch
import logging
from aflora.injection import get_aflora_layers
from federation.hf_hub import HFParameterServer

logger = logging.getLogger(__name__)

class FederatedClient:
    def __init__(self, client_id, model, config, checkpoint_dir="checkpoints"):
        self.client_id = client_id
        self.model = model
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        
        hub_cfg = config.get("federation", {}).get("hub", {})
        if "repo_id" in hub_cfg:
            self.hf_server = HFParameterServer(
                repo_id=hub_cfg["repo_id"], 
                repo_type=hub_cfg.get("repo_type", "dataset")
            )
        else:
            self.hf_server = None
            
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def register_client(self):
        logger.info(f"Client {self.client_id} registered.")
        
    def receive_global_A(self, round_num):
        """
        Downloads and loads the global A matrices from the HF Hub for the given round.
        Returns True if successful, False if the global weights are not ready yet.
        """
        if not self.hf_server:
            logger.warning("HF Hub not configured. Skipping global A sync.")
            return False
            
        global_tensors = self.hf_server.download_global_A_matrices(round_num)
        if global_tensors is None:
            return False
            
        layers = get_aflora_layers(self.model)
        for i, a_tensor in enumerate(global_tensors):
            if i < len(layers):
                layers[i].load_global_A(a_tensor)
        return True
                
    def export_A_update(self, round_num, data_size: int = 0):
        """
        Exports the A matrices from the local model and pushes them to the HF Hub.
        
        Args:
            round_num: Current federation round number.
            data_size: Number of training samples this client used (for weighted FedAvg).
        """
        layers = get_aflora_layers(self.model)
        updates = []
        for layer in layers:
            a_tensor = layer.export_A()
            updates.append(a_tensor.cpu())
            
        if self.hf_server:
            self.hf_server.upload_local_A_matrices(round_num, self.client_id, updates, data_size=data_size)
        else:
            logger.warning("HF Hub not configured. Update not synced.")
            
        return updates
        
    def save_local_adapter(self):
        """
        Saves B and Lambda matrices locally. They never leave the device.
        """
        layers = get_aflora_layers(self.model)
        b_state = {}
        lambda_state = {}
        
        for i, layer in enumerate(layers):
            b_state[f"layer_{i}"] = layer.B.detach().cpu()
            lambda_state[f"layer_{i}"] = layer.Lambda.detach().cpu()
            
        torch.save(b_state, os.path.join(self.checkpoint_dir, "local_B.pt"))
        torch.save(lambda_state, os.path.join(self.checkpoint_dir, "local_lambda.pt"))
        logger.info("Saved local adapter weights (B and Lambda).")
        
    def load_local_adapter(self):
        b_path = os.path.join(self.checkpoint_dir, "local_B.pt")
        lambda_path = os.path.join(self.checkpoint_dir, "local_lambda.pt")
        
        if os.path.exists(b_path) and os.path.exists(lambda_path):
            b_state = torch.load(b_path, weights_only=True)
            lambda_state = torch.load(lambda_path, weights_only=True)
            
            layers = get_aflora_layers(self.model)
            for i, layer in enumerate(layers):
                with torch.no_grad():
                    layer.B.copy_(b_state[f"layer_{i}"].to(layer.B.device))
                    layer.Lambda.copy_(lambda_state[f"layer_{i}"].to(layer.Lambda.device))
            logger.info("Loaded local adapter weights.")
        else:
            logger.info("No local adapter weights found, starting fresh.")
