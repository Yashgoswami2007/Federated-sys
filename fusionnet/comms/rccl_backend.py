"""
FusionNet RCCL Backend — ROCm Collective Communications Library
===============================================================
Wraps PyTorch's distributed API for AMD ROCm / NCCL-compatible AllReduce-based
aggregation between FL nodes.

On AMD ROCm builds, PyTorch's ``torch.distributed`` automatically maps the
``"nccl"`` backend string to RCCL (ROCm Collective Communications Library),
the AMD equivalent of NVIDIA NCCL. The same code runs on both hardware stacks
with no changes.

Usage (single-node multi-GPU, e.g. 2 AMD GPUs):
    # Terminal 1:
    MASTER_ADDR=localhost MASTER_PORT=29500 \\
        python -m torch.distributed.launch --nproc_per_node=2 rccl_demo.py

Usage (multi-node, 2 VMs):
    # On coordinator VM:
    MASTER_ADDR=<coordinator_ip> MASTER_PORT=29500 \\
        python fusionnet/comms/rccl_backend.py --rank 0 --world-size 2
    # On client VM:
    MASTER_ADDR=<coordinator_ip> MASTER_PORT=29500 \\
        python fusionnet/comms/rccl_backend.py --rank 1 --world-size 2
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _is_rocm() -> bool:
    """Returns True if PyTorch was compiled with AMD ROCm (HIP) support."""
    return getattr(torch.version, "hip", None) is not None


class RCCLBackend:
    """
    AMD RCCL (/ NVIDIA NCCL) collective communications backend for FusionNet.

    Provides AllReduce, Broadcast, and Barrier primitives used by the secure
    aggregation layer.  The ``fedavg_allreduce`` method implements data-size
    weighted FedAvg entirely in-collective — no coordinator sees plaintext
    per-client updates.

    Args:
        world_size:   Total number of participating processes (nodes × GPUs).
        rank:         This process's rank in [0, world_size).
        master_addr:  IP / hostname of the rank-0 coordinator.
        master_port:  TCP port for the rendezvous.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        master_addr: str = "localhost",
        master_port: int = 29500,
    ) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be ≥ 1, got {world_size}")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank {rank} out of range [0, {world_size})")

        self.world_size = world_size
        self.rank = rank
        self.master_addr = master_addr
        self.master_port = master_port
        self._initialized = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        """'rccl' on AMD ROCm builds, 'nccl' on NVIDIA CUDA builds."""
        return "rccl" if _is_rocm() else "nccl"

    @property
    def is_coordinator(self) -> bool:
        """True if this process is rank 0 (the aggregation coordinator)."""
        return self.rank == 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def init(self) -> None:
        """
        Initialise the RCCL/NCCL process group.

        Sets ``MASTER_ADDR`` and ``MASTER_PORT`` environment variables, then
        calls ``dist.init_process_group`` with ``backend="nccl"``.  On ROCm
        builds, PyTorch automatically routes this to RCCL.

        Raises:
            RuntimeError: If no GPU is detected.
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "RCCLBackend requires a GPU (AMD ROCm or NVIDIA CUDA). "
                "No CUDA/ROCm-capable device detected on this machine."
            )

        os.environ.setdefault("MASTER_ADDR", self.master_addr)
        os.environ.setdefault("MASTER_PORT", str(self.master_port))

        hw_info = (
            f"AMD ROCm ({torch.version.hip})"
            if _is_rocm()
            else f"NVIDIA CUDA ({torch.version.cuda})"
        )
        logger.info(
            f"Initialising {self.backend_name.upper()} process group | "
            f"rank={self.rank}/{self.world_size} | hardware={hw_info}"
        )

        dist.init_process_group(
            backend="nccl",         # PyTorch auto-maps → RCCL on ROCm builds
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
        )

        self._initialized = True
        logger.info(
            f"{self.backend_name.upper()} process group ready. "
            f"Device: {torch.cuda.get_device_name(0)}"
        )

    def shutdown(self) -> None:
        """Destroy the process group and release resources."""
        if self._initialized:
            dist.destroy_process_group()
            self._initialized = False
            logger.info("RCCL process group destroyed.")

    def __enter__(self) -> "RCCLBackend":
        self.init()
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()

    # ── Collective Operations ─────────────────────────────────────────────────

    def _check_init(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "RCCLBackend not initialised. Call init() or use as context manager."
            )

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: str = "sum",
    ) -> torch.Tensor:
        """
        AllReduce ``tensor`` across all ranks (in-place).

        Args:
            tensor: GPU tensor to reduce. Must be on a CUDA/ROCm device.
            op:     ``"sum"`` or ``"avg"``.

        Returns:
            The reduced tensor (same object, modified in-place).
        """
        self._check_init()
        if op == "sum":
            reduce_op = dist.ReduceOp.SUM
        elif op == "avg":
            reduce_op = dist.ReduceOp.AVG
        else:
            raise ValueError(f"Unsupported op '{op}'. Use 'sum' or 'avg'.")

        dist.all_reduce(tensor, op=reduce_op)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """
        Broadcast ``tensor`` from rank ``src`` to all other ranks (in-place).

        Returns:
            The broadcast tensor.
        """
        self._check_init()
        dist.broadcast(tensor, src=src)
        return tensor

    def barrier(self) -> None:
        """Synchronisation barrier — blocks until all ranks reach this point."""
        self._check_init()
        dist.barrier()

    def fedavg_allreduce(
        self,
        local_tensor: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Data-size weighted FedAvg via AllReduce — no plaintext per-client views.

        Each rank scales its local update by ``num_samples / total_samples``,
        then AllReduce sums the weighted tensors.  The result is the globally
        weighted average, equivalent to FedAvg but computed entirely in-collective.

        Privacy property: The coordinator (rank 0) never sees the raw local
        tensor from any individual client — it only ever observes the aggregate.

        Args:
            local_tensor: This rank's local LoRA adapter update tensor.
            num_samples:  Number of training samples this rank used.

        Returns:
            The globally weighted-averaged tensor (federated global update).
        """
        self._check_init()

        device = local_tensor.device

        # Step 1: AllReduce total sample count so every rank can compute weights
        sample_count = torch.tensor(
            [float(num_samples)], dtype=torch.float64, device=device
        )
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
        total_samples = sample_count.item()

        if total_samples == 0:
            raise ValueError("Total samples across all ranks is 0 — nothing to aggregate.")

        # Step 2: Weight this rank's tensor by its data proportion
        weight = num_samples / total_samples
        weighted = local_tensor.to(torch.float64) * weight

        # Step 3: AllReduce (sum) — produces the weighted average
        dist.all_reduce(weighted, op=dist.ReduceOp.SUM)

        # Return in original dtype
        return weighted.to(local_tensor.dtype)


# ── Convenience Factory ────────────────────────────────────────────────────────

def init_rccl(
    world_size: int,
    rank: int,
    master_addr: str = "localhost",
    master_port: int = 29500,
) -> RCCLBackend:
    """Create and initialise an RCCLBackend in one call."""
    backend = RCCLBackend(world_size, rank, master_addr, master_port)
    backend.init()
    return backend


# ── CLI: Quick Connectivity Test ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="FusionNet RCCL connectivity test")
    parser.add_argument("--rank",       type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-addr", default=os.getenv("MASTER_ADDR", "localhost"))
    parser.add_argument("--master-port", type=int, default=int(os.getenv("MASTER_PORT", "29500")))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with RCCLBackend(args.world_size, args.rank, args.master_addr, args.master_port) as backend:
        device = torch.device("cuda:0")

        # --- AllReduce bandwidth benchmark ---
        sizes = [1_000, 100_000, 10_000_000]  # 1K, 100K, 10M elements
        for n in sizes:
            t = torch.ones(n, dtype=torch.float32, device=device)
            backend.barrier()
            t0 = time.perf_counter()
            result = backend.all_reduce(t, op="sum")
            backend.barrier()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            bandwidth_gb = (n * 4 * 2) / (elapsed_ms / 1000) / 1e9  # 4 bytes/elem, 2× for reduce+broadcast
            logger.info(
                f"AllReduce | n={n:>10,} floats | {elapsed_ms:6.2f} ms | "
                f"{bandwidth_gb:.2f} GB/s | mean={result.mean().item():.1f} (expected={float(args.world_size):.1f})"
            )

        # --- FedAvg AllReduce correctness check ---
        local = torch.full((1000,), fill_value=float(args.rank + 1), device=device)
        num_samples = (args.rank + 1) * 100   # Different data sizes per rank
        fedavg_result = backend.fedavg_allreduce(local, num_samples)
        logger.info(
            f"FedAvg AllReduce | rank={args.rank} | num_samples={num_samples} | "
            f"result_mean={fedavg_result.mean().item():.4f}"
        )

    logger.info("RCCL test complete.")
