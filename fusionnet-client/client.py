import logging
import yaml
import os
import sys
import torch

logger = logging.getLogger(__name__)
from models.loader import load_llama
from aflora.injection import inject_aflora, get_aflora_layers
from federation.client import FederatedClient
from fl_datasets.loader import get_dataset
from training.engine import setup_training, train_local_epoch
from federation.privacy import setup_privacy, PrivacyAccountant


class FusionNetClient:
    def __init__(self, config_path="config.yaml", client_id: int = 0):
        # Validate config with Pydantic schema (falls back to raw YAML if pydantic unavailable)
        try:
            from config_schema import load_and_validate_config
            self.config = load_and_validate_config(config_path)
        except ImportError:
            logger.warning("pydantic not installed — skipping config validation. "
                           "Install with: pip install pydantic")
            import yaml
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

        self.client_id = client_id
        logger.info(f"Initializing FusionNet Client (node {client_id})...")
        self.backend = None  # Will be set by main.py
        
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
        logger.info(f"Using AFLoRA Rank: {self.rank}")

        # Inject AFLoRA adapters
        target_modules = self.config["federation"].get("target_modules", ["q_proj", "v_proj"])
        injected = inject_aflora(self.model, target_modules, self.rank)
        logger.info(f"Injected AFLoRA into {injected} modules.")
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

        # Privacy budget tracking across rounds
        privacy_budget_cap = self.config.get("privacy", {}).get("total_budget_cap", 10.0)
        self.privacy_accountant = PrivacyAccountant(
            budget_cap=privacy_budget_cap,
            checkpoint_dir=self.fed_client.checkpoint_dir,
        )

    def train(self, num_clients: int = 10, round_num: int = 1) -> int:
        import time

        # Check privacy budget before training
        if not self.privacy_accountant.has_budget():
            logger.error("Privacy budget exhausted. Refusing to train.")
            raise RuntimeError(
                f"Privacy budget exhausted (ε_total = {self.privacy_accountant.cumulative_epsilon:.4f}, "
                f"cap = {self.privacy_accountant.budget_cap}). No more training allowed."
            )

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
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            start_time = time.time()
            loss = train_local_epoch(
                self.model, dataloader, optimizer, self.device,
                self.config["federation"], privacy_engine,
            )
            duration = time.time() - start_time
            logger.info(f"Epoch {epoch + 1} done. Avg Loss: {loss:.4f}")
            
            # Report metrics
            if self.backend and self.backend.enabled:
                epsilon = 0.0
                if privacy_engine:
                    try:
                        epsilon = privacy_engine.get_epsilon(delta=1e-5)
                    except Exception as e:
                        logger.warning(f"Failed to get epsilon from privacy engine: {e}")
                
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
        self.last_data_size = len(train_dataset)

        # Record privacy budget consumed in this round
        round_epsilon = 0.0
        if privacy_engine:
            try:
                round_epsilon = privacy_engine.get_epsilon(delta=self.config["privacy"].get("delta", 1e-5))
            except Exception as e:
                logger.warning(f"Could not read epsilon for accounting: {e}")
                round_epsilon = self.config["privacy"].get("epsilon", 1.0)  # Use target as estimate
        self.privacy_accountant.record_round(round_num, round_epsilon)

        return self.last_data_size
