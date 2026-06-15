import torch
from torch.optim import AdamW
from tqdm import tqdm
from .hardware_utils import detect_hardware
from .dp_sgd import setup_dp_engine, get_privacy_metrics

class LoRATrainer:
    def __init__(self, model, learning_rate=2e-4):
        self.model = model
        
        # Hardware Detection & Fallback
        self.hw_config = detect_hardware()
        self.device = self.hw_config['device']
        self.batch_size = self.hw_config['batch_size']
        self.lora_rank = self.hw_config['lora_rank']
        
        print(f"Initialized LoRATrainer on {self.device.upper()} | Batch: {self.batch_size} | Rank: {self.lora_rank}")

        # Initialize Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Privacy engine state
        self.privacy_engine = None

    def enable_dp(self, data_loader, target_epsilon=1.0, target_delta=1e-5, max_grad_norm=1.0, epochs=1):
        """
        Wraps the model, optimizer, and dataloader with the Opacus PrivacyEngine.
        """
        print(f"Enabling DP-SGD (epsilon={target_epsilon}, delta={target_delta}, clip={max_grad_norm})")
        self.model.train()
        
        self.model, self.optimizer, dp_data_loader, self.privacy_engine = setup_dp_engine(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=data_loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
            epochs=epochs
        )
        return dp_data_loader

    def train_epoch(self, data_loader):
        """
        Runs one epoch of training over the given dataloader.
        """
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(data_loader, desc="Training (Local Round)")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass (computes per-sample gradients and adds noise if DP is enabled via Opacus)
            loss.backward()
            
            # Optimizer step
            self.optimizer.step()
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch complete. Average Loss: {avg_loss:.4f}")
        
        if self.privacy_engine:
            # Assuming target_delta was passed, we can estimate it based on len(dataset).
            # For logging purposes, we'll fetch the target delta or use a default.
            target_delta = 1 / len(data_loader.dataset) if hasattr(data_loader, 'dataset') and len(data_loader.dataset) > 0 else 1e-5
            metrics = get_privacy_metrics(self.privacy_engine, target_delta)
            print(f"Privacy Budget Expended: ε = {metrics['epsilon']:.4f}, δ = {metrics['delta']}")
            
        return avg_loss

    def train_step(self, batch):
        # Stub from previous implementation, replaced by train_epoch logic for full dataset iteration.
        pass
