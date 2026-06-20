import torch.nn as nn
from aflora.layer import AFLoRALayer

def inject_aflora(model, target_modules, rank, alpha=16):
    """
    Recursively replaces target modules in the model with AFLoRALayer.
    """
    injected_count = 0
    for name, module in model.named_children():
        # Check if this module matches target modules (e.g. q_proj, v_proj)
        if any(t in name for t in target_modules) and hasattr(module, "weight"):
            aflora_layer = AFLoRALayer(module, rank=rank, alpha=alpha)
            if isinstance(model, nn.ModuleList):
                model[int(name)] = aflora_layer
            elif isinstance(model, nn.ModuleDict):
                model[name] = aflora_layer
            else:
                setattr(model, name, aflora_layer)
            injected_count += 1
        else:
            injected_count += inject_aflora(module, target_modules, rank, alpha)
    return injected_count

def get_aflora_parameters(model):
    """
    Returns lists of A, B and Lambda parameters for the optimizer.
    """
    a_params = []
    b_params = []
    lambda_params = []
    
    for module in model.modules():
        if isinstance(module, AFLoRALayer):
            a_params.append(module.A)
            b_params.append(module.B)
            lambda_params.append(module.Lambda)
            
    return a_params, b_params, lambda_params

def get_aflora_layers(model):
    """
    Returns a list of all AFLoRALayer modules in the model.
    """
    layers = []
    for module in model.modules():
        if isinstance(module, AFLoRALayer):
            layers.append(module)
    return layers
