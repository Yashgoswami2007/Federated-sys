"""
FusionNet Real FL Demo — End-to-End Federated Learning Launcher
================================================================
Launches a complete FL round using your actual hardware setup:

  Node 1 (RTX 5060)  → CUDA client, fast training, generates real loss curves
  Node 2 (AMD 4GB)   → ROCm client, AMD hardware evidence
  Node 3 (CPU only)  → CPU client, proves heterogeneous device support

Usage — all 3 nodes on one machine (simulation):
    python scripts/run_real_fl_demo.py --num-clients 3 --rounds 5

Usage — real multi-machine (Node 1 = coordinator + client 0):
    # On Node 1 (RTX 5060):
    python scripts/hf_coordinator.py --num-clients 3 --rounds 5 &
    python fusionnet-client/main.py --client-id 0 --num-clients 3 --rounds 5

    # On Node 2 (AMD 4GB):
    python fusionnet-client/main.py --client-id 1 --num-clients 3 --rounds 5

    # On Node 3 (CPU):
    python fusionnet-client/main.py --client-id 2 --num-clients 3 --rounds 5

After running, find results at:
    experiments/mvp_sentiment/results/metrics.json  ← real per-round metrics
    experiments/benchmarks/convergence.svg          ← real convergence chart
    docs/training_evidence/                         ← logs for judges
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fl_demo")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _REPO_ROOT / "experiments" / "mvp_sentiment" / "results"
_EVIDENCE_DIR = _REPO_ROOT / "docs" / "training_evidence"
_BENCHMARK_DIR = _REPO_ROOT / "experiments" / "benchmarks"


def print_banner() -> None:
    logger.info("=" * 65)
    logger.info("  FusionNet — Real Federated Learning Demo")
    logger.info("  AMD Developer Hackathon ACT II")
    logger.info("=" * 65)


def detect_hardware() -> dict:
    """Detect available GPU/CPU resources across this machine."""
    info = {"nodes": [], "has_gpu": False, "has_amd": False}

    try:
        import torch
        info["pytorch"] = torch.__version__
        info["has_gpu"] = torch.cuda.is_available()
        info["has_amd"] = getattr(torch.version, "hip", None) is not None

        if info["has_gpu"]:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024 ** 3)
                info["nodes"].append({
                    "gpu_id": i,
                    "name": props.name,
                    "vram_gb": round(vram_gb, 1),
                    "backend": "rocm" if info["has_amd"] else "cuda",
                })
        else:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            info["nodes"].append({"gpu_id": None, "name": "CPU only", "ram_gb": round(ram_gb, 1)})
    except ImportError as e:
        logger.warning(f"Hardware detection partial: {e}")

    return info


def print_hardware_summary(hw: dict) -> None:
    logger.info("")
    logger.info("Hardware detected on this machine:")
    for node in hw["nodes"]:
        if node.get("gpu_id") is not None:
            backend = node["backend"].upper()
            logger.info(f"  GPU [{node['gpu_id']}]: {node['name']} | {node['vram_gb']} GB | {backend}")
        else:
            logger.info(f"  CPU only: {node.get('ram_gb', '?')} GB RAM")

    if hw.get("has_amd"):
        logger.info("  ✓ AMD ROCm backend detected — this run generates AMD evidence!")
    logger.info("")


def run_local_mvp(rounds: int, num_clients: int, report_backend: bool, backend_url: str) -> dict:
    """
    Run the local MVP simulation with real FL round mechanics.
    Uses the existing run_mvp.py infrastructure.
    """
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "experiments" / "mvp_sentiment" / "run_mvp.py"),
        "--rounds", str(rounds),
        "--adapter-rows", "2048",   # Match TinyLlama hidden size
        "--adapter-cols", "8",      # LoRA rank 8
    ]
    if report_backend:
        cmd += ["--report-backend", "--backend-url", backend_url]

    logger.info(f"Running: {' '.join(cmd[1:])}")
    logger.info("")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        logger.error(f"MVP run failed with exit code {result.returncode}")
        sys.exit(1)

    logger.info(f"\nMVP simulation completed in {elapsed:.1f}s")
    return {"elapsed_s": elapsed, "rounds": rounds, "clients": num_clients}


def plot_convergence() -> None:
    """Regenerate the convergence SVG from updated metrics.json."""
    plot_script = _BENCHMARK_DIR / "plot_convergence.py"
    if not plot_script.exists():
        logger.warning("plot_convergence.py not found — skipping chart generation.")
        return
    metrics_path = _RESULTS_DIR / "metrics.json"
    if not metrics_path.exists():
        logger.warning("metrics.json not found — run FL rounds first.")
        return

    cmd = [sys.executable, str(plot_script)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("Convergence chart updated: experiments/benchmarks/convergence.svg")
    else:
        logger.warning(f"Chart generation failed: {result.stderr}")


def save_training_evidence(hw: dict, run_info: dict) -> None:
    """Save hardware info and run summary as judge-visible evidence."""
    _EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    evidence = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hardware": hw,
        "run": run_info,
        "pytorch_version": hw.get("pytorch", "unknown"),
    }

    evidence_path = _EVIDENCE_DIR / "run_summary.json"
    evidence_path.write_text(json.dumps(evidence, indent=2))
    logger.info(f"Hardware evidence saved: {evidence_path.relative_to(_REPO_ROOT)}")

    # Friendly text summary for judges
    summary_lines = [
        "FusionNet Training Evidence",
        "=" * 40,
        f"Timestamp  : {evidence['timestamp']}",
        f"PyTorch    : {evidence['pytorch_version']}",
        f"AMD ROCm   : {'Yes — ' + hw.get('pytorch', '') if hw.get('has_amd') else 'No (CUDA/CPU)'}",
        f"Rounds     : {run_info['rounds']}",
        f"Clients    : {run_info['clients']}",
        f"Duration   : {run_info['elapsed_s']:.1f}s",
        "",
        "Nodes:",
    ]
    for node in hw["nodes"]:
        if node.get("gpu_id") is not None:
            summary_lines.append(
                f"  GPU [{node['gpu_id']}]: {node['name']} "
                f"({node['vram_gb']} GB, {node['backend'].upper()})"
            )
        else:
            summary_lines.append(f"  CPU: {node.get('ram_gb', '?')} GB RAM")

    summary_path = _EVIDENCE_DIR / "run_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    logger.info(f"Text summary saved: {summary_path.relative_to(_REPO_ROOT)}")


def print_final_summary(metrics_path: Path) -> None:
    """Print a judge-friendly results table."""
    if not metrics_path.exists():
        logger.warning("No metrics.json found — run failed or rounds not complete.")
        return

    try:
        metrics = json.loads(metrics_path.read_text())
    except Exception as e:
        logger.error(f"Could not read metrics.json: {e}")
        return

    logger.info("")
    logger.info("=" * 65)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 65)
    logger.info(f"{'Round':>6} | {'Loss':>10} | {'Accuracy':>10} | {'ε spent':>10} | {'Clients':>7}")
    logger.info("-" * 55)

    for m in metrics:
        logger.info(
            f"{m['round']:>6} | {m['avg_loss']:>10.4f} | "
            f"{m['accuracy'] * 100:>9.2f}% | {m['epsilon_max']:>10.4f} | {m['clients']:>7}"
        )

    if metrics:
        first = metrics[0]
        last = metrics[-1]
        accuracy_gain = (last["accuracy"] - first["accuracy"]) * 100
        total_epsilon = last["epsilon_max"]
        logger.info("-" * 55)
        logger.info(f"Accuracy improvement: {accuracy_gain:+.2f}% over {len(metrics)} rounds")
        logger.info(f"Total ε consumed    : {total_epsilon:.4f} (budget cap: 10.0)")
        logger.info(f"Convergence chart   : experiments/benchmarks/convergence.svg")
        logger.info("=" * 65)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FusionNet end-to-end FL demo")
    parser.add_argument("--rounds",       type=int, default=5)
    parser.add_argument("--num-clients",  type=int, default=3)
    parser.add_argument("--report-backend", action="store_true",
                        help="Report metrics to FastAPI backend (requires uvicorn running)")
    parser.add_argument("--backend-url",  default=os.getenv("BACKEND_URL", "http://localhost:8000"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_banner()

    hw = detect_hardware()
    print_hardware_summary(hw)

    run_info = run_local_mvp(args.rounds, args.num_clients, args.report_backend, args.backend_url)
    plot_convergence()
    save_training_evidence(hw, run_info)
    print_final_summary(_RESULTS_DIR / "metrics.json")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Check docs/training_evidence/ for hardware evidence files")
    logger.info("  2. Open experiments/benchmarks/convergence.svg in your browser")
    logger.info("  3. If on AMD hardware, run: scripts/rccl_demo.py")
    logger.info("  4. Start the backend + frontend dashboard for the live demo")


if __name__ == "__main__":
    main()
