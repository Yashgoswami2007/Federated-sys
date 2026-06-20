import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from aflora.injection import get_aflora_parameters
from federation.privacy import setup_privacy, CustomPrivacyEngine

logger = logging.getLogger(__name__)

def train_local_epoch(model, dataloader, optimizer, device, config, privacy_engine=None):
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Local Training")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}

        # LlamaForCausalLM expects labels == input_ids (shifted internally for CLM loss).
        # The banking77 dataset 'labels' column contains class integers (0-76), which
        # causes a shape mismatch. Override with input_ids for self-supervised training.
        batch['labels'] = batch['input_ids'].clone()

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        
        if privacy_engine:
            if isinstance(privacy_engine, CustomPrivacyEngine):
                # CustomPrivacyEngine.step() clips gradients, adds noise,
                # and calls optimizer.step() internally.
                privacy_engine.step()
            else:
                # Opacus wraps the optimizer during make_private_with_epsilon(),
                # so stepping the optimizer triggers DP-SGD (per-sample gradient
                # clipping + calibrated noise) automatically.
                optimizer.step()
        else:
            optimizer.step()
            
        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f"Epoch complete. Average loss: {avg_loss:.4f} over {num_batches} batches")
    return avg_loss

def setup_training(model, train_dataset, config):
    batch_size = config.get("batch_size", 4)
    lr = config.get("learning_rate", 1e-4)
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    a_params, b_params, lambda_params = get_aflora_parameters(model)
    optimizer = torch.optim.AdamW([
        {'params': a_params, 'lr': lr},
        {'params': b_params, 'lr': lr},
        {'params': lambda_params, 'lr': lr}
    ])
    
    return dataloader, optimizer
