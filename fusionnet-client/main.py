import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    # Authenticate with HF before anything else
    from auth import hf_login
    hf_login()

    from client import FusionNetClient
    from backend_client import BackendClient

    parser = argparse.ArgumentParser(description="FusionNet local client node")
    parser.add_argument("--client-id",   type=int, default=0,  help="Unique integer ID for this node")
    parser.add_argument("--num-clients", type=int, default=10, help="Total federation size")
    parser.add_argument("--rounds",      type=int, default=1,  help="Number of FL rounds to run")
    args = parser.parse_args()

    client = FusionNetClient("config.yaml", client_id=args.client_id)
    
    # Init backend client
    backend_url = client.config.get("backend", {}).get("url", "http://localhost:8000")
    backend_enabled = client.config.get("backend", {}).get("enabled", True)
    backend = BackendClient(backend_url, os.getenv("HF_TOKEN"))
    backend.enabled = backend_enabled
    client.backend = backend  # Attach to client for use in training loop
    
    if backend.enabled:
        backend.register(f"client_{args.client_id}", client.device_profile_info)
        backend.start_heartbeat_loop(f"client_{args.client_id}", client.config.get("backend", {}).get("heartbeat_interval", 30))
        backend.report_event("device.registered", f"Node client_{args.client_id} joined the federation", "info")
    
    client.fed_client.register_client()

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'='*50}")
        print(f"  FEDERATED ROUND {round_num}/{args.rounds}")
        print(f"{'='*50}")

        # Step 1: pull latest global A matrices (skip on round 1 — none exist yet)
        if round_num > 1:
            print(f"[Round {round_num}] Downloading global A matrices...")
            success = client.fed_client.receive_global_A(round_num - 1)
            if not success:
                print(f"[Round {round_num}] No global weights found — continuing with local state.")

        # Step 2: local training
        print(f"[Round {round_num}] Starting local training...")
        client.train(num_clients=args.num_clients, round_num=round_num)

        # Step 3: push local A update to HF Hub
        print(f"[Round {round_num}] Exporting A matrices to HF Hub...")
        updates = client.fed_client.export_A_update(round_num=round_num)
        print(f"[Round {round_num}] Pushed {len(updates)} A matrices.")
        if backend.enabled:
            backend.update_round(round_num, received_clients=1)  # Simplified: coordinator should really track this
            backend.report_event("client.uploaded", f"Client {args.client_id} uploaded weights for round {round_num}")

    print("\nAll rounds complete. FusionNet client finished.")
