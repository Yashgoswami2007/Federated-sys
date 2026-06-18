import yaml
import os
import sys
import torch
from models.loader import load_llama
from aflora.injection import inject_aflora, get_aflora_layers
from federation.client import FederatedClient
from fl_datasets.loader import get_dataset
from training.engine import setup_training, train_local_epoch
from federation.privacy import setup_privacy


class FusionNetClient:
    def __init__(self, config_path="config.yaml", client_id: int = 0):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.client_id = client_id
        print(f"Initializing FusionNet Client (node {client_id})...")
        self.backend = None # Will be set by main.py
        
        # Determine device profile info for registration
        self.device_profile_info = {
            "os": os.name,
            "python_version": sys.version.split(" ")[0],
            "hardware_tier": "cpu"
        }

        # Load model
        self.model, self.tokenizer, self.device_profile = load_llama(
            self.config["model"]["name"],
            self.config["model"].get("quantization_type", "nf4"),
        )

        # Determine LoRA rank from device profile
        profile_config = self.config["device_profiles"].get(self.device_profile, {})
        self.rank = profile_config.get("rank", self.config["federation"]["lora_rank"])
        print(f"Using AFLoRA Rank: {self.rank}")

        # Inject AFLoRA adapters
        target_modules = self.config["federation"].get("target_modules", ["q_proj", "v_proj"])
        injected = inject_aflora(self.model, target_modules, self.rank)
        print(f"Injected AFLoRA into {injected} modules.")
        self.device_profile_info["hardware_tier"] = self.device_profile
        self.device_profile_info["lora_rank"] = self.rank

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )

        # Federation client — use client_id as unique upload key
        self.fed_client = FederatedClient(
            client_id=f"client_{client_id}",
            model=self.model,
            config=self.config,
        )
        self.fed_client.load_local_adapter()

    def train(self, num_clients: int = 10, round_num: int = 1):
        import time
        train_dataset, _ = get_dataset(
            self.config["dataset"],
            self.tokenizer,
            device_tier=self.device_profile,
            client_id=self.client_id,
            num_clients=num_clients,
        )
        dataloader, optimizer = setup_training(self.model, train_dataset, self.config["federation"])

        self.model, optimizer, dataloader, privacy_engine = setup_privacy(
            self.model, optimizer, dataloader, self.config["privacy"]
        )

        epochs = self.config["federation"].get("local_epochs", 1)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            start_time = time.time()
            loss = train_local_epoch(
                self.model, dataloader, optimizer, self.device,
                self.config["federation"], privacy_engine,
            )
            duration = time.time() - start_time
            print(f"Epoch {epoch + 1} done. Avg Loss: {loss:.4f}")
            
            # Report metrics
            if self.backend and self.backend.enabled:
                epsilon = 0.0
                if privacy_engine:
                    try:
                        epsilon = privacy_engine.get_epsilon(delta=1e-5)
                    except:
                        pass
                
                self.backend.report_metrics(
                    client_id=f"client_{self.client_id}",
                    round_num=round_num,
                    epoch=epoch + 1,
                    metrics={
                        "avg_loss": float(loss),
                        "training_duration_s": duration,
                        "data_size": len(train_dataset),
                        "epsilon_spent": float(epsilon)
                    }
                )

        self.fed_client.save_local_adapter()
