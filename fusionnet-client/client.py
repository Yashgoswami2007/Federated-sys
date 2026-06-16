import yaml
import os
import torch
from models.loader import load_llama
from aflora.injection import inject_aflora, get_aflora_layers
from federation.client import FederatedClient
from fl_datasets.loader import get_dataset
from training.engine import setup_training, train_local_epoch
from federation.privacy import setup_privacy

class FusionNetClient:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        print("Initializing FusionNet Client...")
        
        # Load Model
        self.model, self.tokenizer, self.device_profile = load_llama(
            self.config["model"]["name"],
            self.config["model"].get("quantization_type", "nf4")
        )
        
        # Determine Rank
        profile_config = self.config["device_profiles"].get(self.device_profile, {})
        self.rank = profile_config.get("rank", self.config["federation"]["lora_rank"])
        print(f"Using AFLoRA Rank: {self.rank}")
        
        # Inject AFLoRA
        target_modules = self.config["federation"].get("target_modules", ["q_proj", "v_proj"])
        injected = inject_aflora(self.model, target_modules, self.rank)
        print(f"Injected AFLoRA into {injected} modules.")
        
        # Determine Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        
        # Initialize Federation Client
        self.fed_client = FederatedClient(
            client_id="fusionnet_node_01",
            model=self.model,
            config=self.config
        )
        self.fed_client.load_local_adapter()
        
    def train(self, client_id: int = 0, num_clients: int = 10):
        train_dataset, _ = get_dataset(
            self.config["dataset"],
            self.tokenizer,
            device_tier=self.device_profile,
            client_id=client_id,
            num_clients=num_clients,
        )
        dataloader, optimizer = setup_training(self.model, train_dataset, self.config["federation"])
        
        self.model, optimizer, dataloader, privacy_engine = setup_privacy(
            self.model, optimizer, dataloader, self.config["privacy"]
        )
        
        epochs = self.config["federation"].get("local_epochs", 1)
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            loss = train_local_epoch(
                self.model, dataloader, optimizer, self.device, 
                self.config["federation"], privacy_engine
            )
            print(f"Epoch {epoch+1} finished. Avg Loss: {loss:.4f}")
            
        self.fed_client.save_local_adapter()
