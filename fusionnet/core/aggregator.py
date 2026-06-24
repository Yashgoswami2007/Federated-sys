import copy
import torch

def _has_inf_or_nan(state_dict):
    """Check if any tensor in the state_dict contains NaN or Inf."""
    for tensor in state_dict.values():
        if not torch.isfinite(tensor).all():
            return True
    return False

def _compute_norm(state_dict):
    """Compute the L2 norm of the entire state_dict."""
    squared_sum = 0.0
    for tensor in state_dict.values():
        squared_sum += torch.sum(tensor.float() ** 2).item()
    return squared_sum ** 0.5

def fed_avg(client_weights_list, client_data_sizes, max_norm_ratio=3.0):
    """
    Computes the weighted average of model weights (FedAvg) with Byzantine Fault Tolerance.
    Filters out updates with NaN/Inf values or excessively large norms to prevent scaling attacks and corruption.
    
    Args:
        client_weights_list: List of state_dicts from each client.
        client_data_sizes: List of integers representing the number of samples per client.
        max_norm_ratio: Maximum allowed ratio of a client's norm compared to the median norm.
        
    Returns:
        A new state_dict containing the averaged weights, or None if no valid weights remain.
    """
    if not client_weights_list or not client_data_sizes:
        return None
        
    if len(client_weights_list) != len(client_data_sizes):
        raise ValueError("Number of weight updates must match number of data sizes.")

    # Defense 1 & 2: Filter out NaN/Inf and compute norms
    valid_weights = []
    valid_sizes = []
    norms = []
    
    for weights, size in zip(client_weights_list, client_data_sizes):
        if not _has_inf_or_nan(weights):
            valid_weights.append(weights)
            valid_sizes.append(size)
            norms.append(_compute_norm(weights))
            
    if not valid_weights:
        return None

    # Defense 3: Filter out excessively large updates (e.g., 1000x scaling attacks)
    median_norm = sorted(norms)[len(norms) // 2]
    filtered_weights = []
    filtered_sizes = []
    
    for w, size, norm in zip(valid_weights, valid_sizes, norms):
        if norm <= median_norm * max_norm_ratio:
            filtered_weights.append(w)
            filtered_sizes.append(size)

    if not filtered_weights:
        return None

    total_data_samples = sum(filtered_sizes)
    
    # Initialize the aggregated weights with the first valid client's scaled weights
    averaged_weights = copy.deepcopy(filtered_weights[0])
    first_client_weight = filtered_sizes[0] / total_data_samples
    
    for key in averaged_weights.keys():
        # Make sure we work with floats to avoid precision issues
        averaged_weights[key] = averaged_weights[key].float() * first_client_weight

    # Iterate over the rest of the valid clients
    for i in range(1, len(filtered_weights)):
        client_weight = filtered_sizes[i] / total_data_samples
        state_dict = filtered_weights[i]
        
        for key in averaged_weights.keys():
            averaged_weights[key] += state_dict[key].float() * client_weight
            
    # Convert back to the original dtype of the first client
    original_dtype = next(iter(filtered_weights[0].values())).dtype
    for key in averaged_weights.keys():
        averaged_weights[key] = averaged_weights[key].to(original_dtype)

    return averaged_weights

def fed_median(client_weights_list):
    """
    Robust aggregation using coordinate-wise median to defend against model poisoning and backdoors.
    
    Args:
        client_weights_list: List of state_dicts from each client.
        
    Returns:
        A new state_dict containing the median weights.
    """
    if not client_weights_list:
        return None
        
    # Filter out invalid updates first
    valid_weights = [w for w in client_weights_list if not _has_inf_or_nan(w)]
    if not valid_weights:
        return None
        
    median_weights = copy.deepcopy(valid_weights[0])
    
    for key in median_weights.keys():
        # Stack all tensors for this key across clients
        stacked_tensors = torch.stack([w[key].float() for w in valid_weights])
        # Compute the median along the client dimension (dim=0)
        median_tensor, _ = torch.median(stacked_tensors, dim=0)
        median_weights[key] = median_tensor
        
    original_dtype = next(iter(valid_weights[0].values())).dtype
    for key in median_weights.keys():
        median_weights[key] = median_weights[key].to(original_dtype)
        
    return median_weights

def secure_aggregate(
    updates: list,
    client_data_sizes: list | None = None,
    method: str = "secure_shares",
    num_servers: int = 3,
) -> dict | None:
    """
    Secure aggregation dispatcher.

    Args:
        updates:           List of client state_dicts.
        client_data_sizes: Number of training samples per client (FedAvg weighting).
        method:            Aggregation strategy:
                             - ``"secure_shares"`` — additive secret sharing MPC (default)
                             - ``"fedavg"``         — plain weighted FedAvg (no MPC)
                             - ``"median"``         — Byzantine-robust coordinate-wise median
        num_servers:       Number of MPC servers (only for ``"secure_shares"``).

    Returns:
        Aggregated state_dict, or None if no valid updates.
    """
    if not updates:
        return None

    if method == "secure_shares":
        from .secure_agg import secure_aggregate as _mpc_aggregate
        return _mpc_aggregate(
            client_updates=updates,
            client_data_sizes=client_data_sizes,
            num_servers=num_servers,
        )
    elif method == "fedavg":
        sizes = client_data_sizes if client_data_sizes else [1] * len(updates)
        return fed_avg(updates, sizes)
    elif method == "median":
        return fed_median(updates)
    else:
        raise ValueError(f"Unknown aggregation method '{method}'. "
                         f"Choose from: 'secure_shares', 'fedavg', 'median'.")

