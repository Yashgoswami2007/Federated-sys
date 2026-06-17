import sys
import os
import argparse
import traceback

# Log everything to a file since PowerShell redirect doesn't capture Python output
_log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../run_output.txt'), 'w', buffering=1)
sys.stdout = _log
sys.stderr = _log

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import FusionNetClient

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="FusionNet example local training")
        parser.add_argument("--client-id",   type=int, default=0,  help="Unique node ID")
        parser.add_argument("--num-clients", type=int, default=10, help="Total federation size")
        args = parser.parse_args()

        print("--- FusionNet Example Local Training ---")
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        client = FusionNetClient(config_path, client_id=args.client_id)

        print(f"\nStarting local training (node {args.client_id}/{args.num_clients})...")
        client.train(num_clients=args.num_clients)

        print("\nTraining completed. Adapters saved to checkpoints/")
    except Exception:
        traceback.print_exc()
    finally:
        _log.close()
