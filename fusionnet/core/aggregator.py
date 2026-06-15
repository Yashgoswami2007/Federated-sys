import copy
import torch

def fed_avg(client_weights_list, client_data_sizes):
    """
    Computes the weighted average of model weights (FedAvg).
    
    Args:
        client_weights_list: List of state_dicts from each client.
        client_data_sizes: List of integers representing the number of samples per client.
        
    Returns:
        A new state_dict containing the averaged weights.
    """
    if not client_weights_list or not client_data_sizes:
        return None
        
    if len(client_weights_list) != len(client_data_sizes):
        raise ValueError("Number of weight updates must match number of data sizes.")

    total_data_samples = sum(client_data_sizes)
    
    # Initialize the aggregated weights with the first client's scaled weights
    averaged_weights = copy.deepcopy(client_weights_list[0])
    first_client_weight = client_data_sizes[0] / total_data_samples
    
    for key in averaged_weights.keys():
        # Make sure we work with floats to avoid precision issues
        averaged_weights[key] = averaged_weights[key].float() * first_client_weight

    # Iterate over the rest of the clients
    for i in range(1, len(client_weights_list)):
        client_weight = client_data_sizes[i] / total_data_samples
        state_dict = client_weights_list[i]
        
        for key in averaged_weights.keys():
            averaged_weights[key] += state_dict[key].float() * client_weight
            
    # Convert back to the original dtype of the first client
    original_dtype = next(iter(client_weights_list[0].values())).dtype
    for key in averaged_weights.keys():
        averaged_weights[key] = averaged_weights[key].to(original_dtype)

    return averaged_weights

def secure_aggregate(updates):
    # Stub for Secure Aggregation (MPC) - to be implemented in the future
    pass
