import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from aflora.injection import get_aflora_parameters
from federation.privacy import setup_privacy, CustomPrivacyEngine

def train_local_epoch(model, dataloader, optimizer, device, config, privacy_engine=None):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Local Training")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        
        if privacy_engine:
            if isinstance(privacy_engine, CustomPrivacyEngine):
                privacy_engine.step()
                optimizer.zero_grad()
            else:
                # Opacus PrivacyEngine step logic
                optimizer.step()
        else:
            optimizer.step()
            
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def setup_training(model, train_dataset, config):
    batch_size = config.get("batch_size", 4)
    lr = config.get("learning_rate", 1e-4)
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    b_params, lambda_params = get_aflora_parameters(model)
    optimizer = torch.optim.AdamW([
        {'params': b_params, 'lr': lr},
        {'params': lambda_params, 'lr': lr}
    ])
    
    return dataloader, optimizer
