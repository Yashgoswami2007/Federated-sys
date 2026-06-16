import os
import json
import torch
from federation.privacy import serialize_tensor_base64, deserialize_tensor_base64
from aflora.injection import get_aflora_layers

class FederatedClient:
    def __init__(self, client_id, model, config, checkpoint_dir="checkpoints"):
        self.client_id = client_id
        self.model = model
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def register_client(self):
        print(f"Client {self.client_id} registered.")
        
    def receive_global_A(self, a_payloads):
        """
        Loads the global A matrices from the coordinator.
        a_payloads: List of JSON/dict objects containing base64 A matrices.
        """
        layers = get_aflora_layers(self.model)
        for i, payload in enumerate(a_payloads):
            if i < len(layers):
                a_tensor = deserialize_tensor_base64(payload["payload"], payload["shape"])
                layers[i].load_global_A(a_tensor)
                
    def export_A_update(self, round_num):
        """
        Exports the A matrices from the local model to send to the coordinator.
        """
        layers = get_aflora_layers(self.model)
        updates = []
        for layer in layers:
            a_tensor = layer.export_A()
            updates.append({
                "client_id": self.client_id,
                "round": round_num,
                "shape": list(a_tensor.shape),
                "dtype": "float16",
                "encoding": "base64",
                "payload": serialize_tensor_base64(a_tensor)
            })
        return updates
        
    def save_local_adapter(self):
        """
        Saves B and Lambda matrices locally. They never leave the device.
        """
        layers = get_aflora_layers(self.model)
        b_state = {}
        lambda_state = {}
        
        for i, layer in enumerate(layers):
            b_state[f"layer_{i}"] = layer.B.detach().cpu()
            lambda_state[f"layer_{i}"] = layer.Lambda.detach().cpu()
            
        torch.save(b_state, os.path.join(self.checkpoint_dir, "local_B.pt"))
        torch.save(lambda_state, os.path.join(self.checkpoint_dir, "local_lambda.pt"))
        print("Saved local adapter weights (B and Lambda).")
        
    def load_local_adapter(self):
        b_path = os.path.join(self.checkpoint_dir, "local_B.pt")
        lambda_path = os.path.join(self.checkpoint_dir, "local_lambda.pt")
        
        if os.path.exists(b_path) and os.path.exists(lambda_path):
            b_state = torch.load(b_path)
            lambda_state = torch.load(lambda_path)
            
            layers = get_aflora_layers(self.model)
            for i, layer in enumerate(layers):
                with torch.no_grad():
                    layer.B.copy_(b_state[f"layer_{i}"].to(layer.B.device))
                    layer.Lambda.copy_(lambda_state[f"layer_{i}"].to(layer.Lambda.device))
            print("Loaded local adapter weights.")
        else:
            print("No local adapter weights found, starting fresh.")
