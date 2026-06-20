import torch
import torch.nn as nn
import math

class AFLoRALayer(nn.Module):
    """
    Adaptive Federated LoRA Layer.
    ΔW = A × Λ × B
    A is the global shared matrix (trained locally, aggregated globally).
    Λ is the local trainable diagonal matrix.
    B is the local trainable matrix.
    """
    def __init__(self, base_layer, rank, alpha=16):
        super().__init__()
        self.base_layer = base_layer
        
        if hasattr(base_layer, "in_features"):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        else:
            # Handle BitsAndBytes Linear4bit
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
            
        self.rank = rank
        self.scaling = alpha / rank
        
        # A: Global Shared Matrix (out_features x rank)
        self.A = nn.Parameter(torch.empty(self.out_features, rank), requires_grad=True)
        # B: Local Trainable Matrix (rank x in_features)
        self.B = nn.Parameter(torch.empty(rank, self.in_features), requires_grad=True)
        # Λ: Local Diagonal Matrix (rank)
        self.Lambda = nn.Parameter(torch.ones(rank), requires_grad=True)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize A with Kaiming uniform (coordinator will overwrite this, but we need an initial state)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # Initialize B with zeros
        nn.init.zeros_(self.B)
        # Initialize Lambda with ones
        nn.init.ones_(self.Lambda)
        
    def load_global_A(self, new_A: torch.Tensor):
        with torch.no_grad():
            self.A.copy_(new_A.to(self.A.device, dtype=self.A.dtype))
            
    def export_A(self):
        return self.A.detach().cpu().clone()
        
    def forward(self, x):
        base_out = self.base_layer(x)

        # x shape: (..., in_features)
        # B shape: (rank, in_features)
        # Lambda shape: (rank)
        # A shape: (out_features, rank)

        # Ensure AFLoRA parameters are on the same device/dtype as the input
        # (needed when bitsandbytes places base layers on CUDA but parameters default to CPU)
        device = x.device
        dtype  = x.dtype
        
        # A is now trainable, so we cast it directly like B and Lambda
        A      = self.A.to(device=device, dtype=dtype)
        
        B      = self.B.to(device=device, dtype=dtype)
        Lambda = self.Lambda.to(device=device, dtype=dtype)

        # lora_B_out = x @ B^T
        lora_B_out = torch.nn.functional.linear(x, B)
        # Multiply by diagonal Lambda
        lora_Lambda_out = lora_B_out * Lambda
        # lora_out = lora_Lambda_out @ A^T
        lora_out = torch.nn.functional.linear(lora_Lambda_out, A)

        return base_out + lora_out * self.scaling

