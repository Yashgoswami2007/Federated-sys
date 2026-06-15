import copy
import torch
from .lora_trainer import LoRATrainer

class FLCoordinator:
    def __init__(self, model):
        """
        Initializes the FL Coordinator.
        
        Args:
            model (PeftModel): The initialized 4-bit model with LoRA adapters.
        """
        self.model = model

    def apply_global_weights(self, global_weights):
        """
        Applies the received global weights to the local LoRA model.
        """
        print("Applying global weights to local model...")
        # Strict=False because we only update the LoRA adapter weights, not the frozen base model.
        self.model.load_state_dict(global_weights, strict=False)

    def extract_local_weights(self):
        """
        Extracts the trainable weights (LoRA adapter) to send to the server.
        """
        print("Extracting local LoRA weights...")
        # Get only the trainable parameters (LoRA)
        state_dict = self.model.state_dict()
        trainable_state_dict = {
            k: v.cpu() for k, v in state_dict.items() if v.requires_grad
        }
        return trainable_state_dict

    def start_round(self, global_weights, local_dataloader, epochs=1, dp_epsilon=1.0, dp_delta=1e-5, max_grad_norm=1.0):
        """
        Executes a single federated learning round locally.
        
        Args:
            global_weights (dict): The global model state_dict received from the server.
            local_dataloader (DataLoader): PyTorch DataLoader containing local data.
            epochs (int): Number of local epochs to train.
            dp_epsilon (float): Target DP epsilon budget.
            dp_delta (float): Target DP delta budget.
            max_grad_norm (float): Gradient clipping norm for DP.
            
        Returns:
            updated_weights (dict): The locally trained LoRA state_dict.
            metrics (dict): Training and privacy metrics.
        """
        print("\n--- Starting Local FL Round ---")
        
        if global_weights:
            self.apply_global_weights(global_weights)

        # Initialize trainer
        trainer = LoRATrainer(model=self.model, learning_rate=2e-4)

        # Enable DP-SGD via Opacus
        # dp_dataloader is a modified dataloader that Opacus uses to properly track batches/noise
        dp_dataloader = trainer.enable_dp(
            data_loader=local_dataloader,
            target_epsilon=dp_epsilon,
            target_delta=dp_delta,
            max_grad_norm=max_grad_norm,
            epochs=epochs
        )

        avg_loss = 0.0
        for epoch in range(epochs):
            print(f"\nLocal Epoch {epoch+1}/{epochs}")
            avg_loss = trainer.train_epoch(dp_dataloader)
            
        # Extract the new local weights
        updated_weights = self.extract_local_weights()
        
        metrics = {
            "loss": avg_loss,
            "data_size": len(local_dataloader.dataset) if hasattr(local_dataloader, 'dataset') else 0
        }
        
        # Add final privacy metrics
        if trainer.privacy_engine:
            from .dp_sgd import get_privacy_metrics
            privacy_metrics = get_privacy_metrics(trainer.privacy_engine, dp_delta)
            metrics.update(privacy_metrics)
            
        print("--- Local FL Round Complete ---")
        return updated_weights, metrics
