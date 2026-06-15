from opacus import PrivacyEngine

def setup_dp_engine(model, optimizer, data_loader, target_epsilon, target_delta, max_grad_norm, epochs):
    """
    Sets up the Opacus Privacy Engine for DP-SGD.
    
    Args:
        model: The PyTorch model (e.g., PEFT LoRA model).
        optimizer: The PyTorch optimizer.
        data_loader: The training PyTorch DataLoader.
        target_epsilon (float): The target privacy budget (epsilon).
        target_delta (float): The target privacy budget (delta), typically 1 / len(dataset).
        max_grad_norm (float): The maximum L2 norm for gradient clipping.
        epochs (int): The total number of training epochs.
        
    Returns:
        model, optimizer, data_loader, privacy_engine
    """
    privacy_engine = PrivacyEngine()

    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )
    
    return model, optimizer, data_loader, privacy_engine

def get_privacy_metrics(privacy_engine, target_delta):
    """
    Returns the current privacy budget spent.
    """
    if privacy_engine:
        epsilon = privacy_engine.get_epsilon(target_delta)
        return {"epsilon": epsilon, "delta": target_delta}
    return {"epsilon": 0.0, "delta": target_delta}

def add_dp_noise(gradients, epsilon, delta):
    # Legacy Stub for manual DP-SGD noise addition, kept for reference
    return gradients
