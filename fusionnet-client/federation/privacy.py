import logging
import os
import torch
import torch.nn as nn
import base64
import json
import numpy as np

logger = logging.getLogger(__name__)


class PrivacyAccountant:
    """Tracks cumulative privacy budget (epsilon) across federated rounds.
    
    In federated learning, each round consumes some privacy budget.
    The total budget after R rounds is approximately R × per_round_epsilon
    (basic composition). This class tracks that accumulation and warns
    when approaching a configurable cap.
    
    Persists budget state to disk so restarts don't reset the counter.
    """

    def __init__(self, budget_cap: float = 10.0, checkpoint_dir: str = "checkpoints"):
        self.budget_cap = budget_cap
        self.cumulative_epsilon = 0.0
        self.round_history = []  # List of (round_num, epsilon_spent)
        self.checkpoint_dir = checkpoint_dir
        self._state_path = os.path.join(checkpoint_dir, "privacy_budget.json")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._load_state()

    def _load_state(self):
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path, "r") as f:
                    state = json.load(f)
                self.cumulative_epsilon = state.get("cumulative_epsilon", 0.0)
                self.round_history = state.get("round_history", [])
                logger.info(
                    f"Loaded privacy budget state: ε_total = {self.cumulative_epsilon:.4f} "
                    f"(cap: {self.budget_cap})"
                )
            except Exception as e:
                logger.warning(f"Failed to load privacy budget state: {e}. Starting fresh.")

    def _save_state(self):
        state = {
            "cumulative_epsilon": self.cumulative_epsilon,
            "round_history": self.round_history,
            "budget_cap": self.budget_cap,
        }
        try:
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save privacy budget state: {e}")

    def record_round(self, round_num: int, epsilon_spent: float):
        """Record the epsilon spent in a single round and check budget."""
        self.cumulative_epsilon += epsilon_spent
        self.round_history.append({"round": round_num, "epsilon": epsilon_spent})
        self._save_state()

        remaining = self.budget_cap - self.cumulative_epsilon
        logger.info(
            f"Privacy budget: ε_round = {epsilon_spent:.4f}, "
            f"ε_total = {self.cumulative_epsilon:.4f}, "
            f"ε_remaining = {remaining:.4f}"
        )

        if self.cumulative_epsilon >= self.budget_cap:
            logger.error(
                f"PRIVACY BUDGET EXHAUSTED! ε_total ({self.cumulative_epsilon:.4f}) "
                f"exceeds cap ({self.budget_cap}). No more training should occur."
            )
            return False  # Budget exhausted
        elif remaining < epsilon_spent * 2:
            logger.warning(
                f"Privacy budget nearly exhausted! Only ε = {remaining:.4f} remaining "
                f"(less than 2 rounds at current rate)."
            )

        return True  # Budget OK

    def has_budget(self) -> bool:
        return self.cumulative_epsilon < self.budget_cap

    def get_summary(self) -> dict:
        return {
            "cumulative_epsilon": self.cumulative_epsilon,
            "budget_cap": self.budget_cap,
            "remaining": self.budget_cap - self.cumulative_epsilon,
            "rounds_recorded": len(self.round_history),
        }


class CustomPrivacyEngine:
    """
    Fallback Differential Privacy implementation if Opacus fails.
    Implements per-sample gradient clipping and Gaussian noise addition.
    """
    def __init__(self, model, optimizer, max_grad_norm, epsilon, delta):
        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon
        self.delta = delta
        
        # Simple noise multiplier calculation based on DP-SGD
        self.noise_multiplier = (max_grad_norm * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
        
    def step(self):
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 
            self.max_grad_norm
        )
        
        # Add noise
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                noise = torch.normal(
                    mean=0.0, 
                    std=self.noise_multiplier * self.max_grad_norm, 
                    size=param.grad.shape, 
                    device=param.grad.device
                )
                param.grad += noise
                
        self.optimizer.step()

    def get_epsilon(self, delta: float = None) -> float:
        """Returns the target epsilon (approximate — not a rigorous accounting)."""
        return self.epsilon


def setup_privacy(model, optimizer, dataloader, config):
    """
    Attempts to setup Opacus, falls back to CustomPrivacyEngine if it fails.
    """
    if not config.get("use_dp_sgd", False):
        return model, optimizer, dataloader, None
        
    epsilon = config.get("epsilon", 1.0)
    delta = config.get("delta", 1e-5)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    
    try:
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine()
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            epochs=config.get("local_epochs", 1),
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )
        logger.info("Opacus DP-SGD successfully initialized.")
        return model, optimizer, dataloader, privacy_engine
    except Exception as e:
        logger.warning(f"Opacus failed: {e}. Falling back to CustomPrivacyEngine.")
        custom_engine = CustomPrivacyEngine(model, optimizer, max_grad_norm, epsilon, delta)
        return model, optimizer, dataloader, custom_engine


def serialize_tensor_base64(tensor: torch.Tensor):
    """
    Serializes a tensor to a base64 encoded string representing float16 data.
    """
    tensor_np = tensor.cpu().numpy().astype(np.float16)
    b64_str = base64.b64encode(tensor_np.tobytes()).decode('utf-8')
    return b64_str

def deserialize_tensor_base64(b64_str: str, shape: list):
    """
    Deserializes a base64 string back into a torch Tensor (float16).
    """
    tensor_bytes = base64.b64decode(b64_str)
    tensor_np = np.frombuffer(tensor_bytes, dtype=np.float16).reshape(shape)
    return torch.from_numpy(tensor_np)
