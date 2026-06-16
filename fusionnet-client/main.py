import argparse
from client import FusionNetClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FusionNet local client node")
    parser.add_argument("--client-id",   type=int, default=0,  help="Unique integer ID for this node (default: 0)")
    parser.add_argument("--num-clients", type=int, default=10, help="Total federation size (default: 10)")
    args = parser.parse_args()

    client = FusionNetClient("config.yaml")
    client.fed_client.register_client()

    print(f"Starting local training round (node {args.client_id}/{args.num_clients})...")
    client.train(client_id=args.client_id, num_clients=args.num_clients)

    print("Exporting A updates...")
    updates = client.fed_client.export_A_update(round_num=1)
    print(f"Exported {len(updates)} A matrices.")
    print("Payload sample:", str(updates[0])[:100] + "...")
    print("FusionNet Client finished.")
