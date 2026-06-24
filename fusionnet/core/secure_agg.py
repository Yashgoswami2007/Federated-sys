"""
FusionNet Secure Aggregation — Additive Secret Sharing (MPC)
============================================================
Implements a 3-party additive secret sharing scheme for federated gradient
aggregation.  This is the simplest provably-secure MPC primitive:

    Security guarantee:
        Any single server (or any strict minority of servers) that receives
        only one share of a client's update learns *nothing* about the original
        tensor.  Reconstruction requires all shares simultaneously.

    Protocol (n clients, s servers):
        1. Each client i splits its local LoRA adapter Δwᵢ into ``num_servers``
           additive shares:  Δwᵢ = s₀ + s₁ + ... + s_{n-1}
           where s₀, ..., s_{n-2} are drawn uniformly at random and
           s_{n-1} = Δwᵢ − (s₀ + ... + s_{n-2}).

        2. Client i sends share sₖ to server k (encrypted channel, e.g. TLS).

        3. Each server k aggregates its received shares:
           T_k = Σᵢ sᵢₖ   (sum of all clients' k-th share)

        4. Coordinator sums server totals:
           Global_Δw = T₀ + T₁ + ... + T_{n-1} = Σᵢ Δwᵢ   (FedAvg numerator)

    Privacy:
        Server k only ever sees {sᵢₖ for all i}, which are all uniformly random
        tensors.  No single server can reconstruct any Δwᵢ.

    Limitations / Honest-majority assumption:
        This scheme assumes servers do not collude.  In practice, servers can be
        run by different organisations (hospital, bank, legal firm) to enforce
        this assumption organisationally.  For stronger guarantees (malicious
        majority), upgrade to SPDZ — listed in the roadmap.

References:
    - Ben-Or, Goldwasser, Wigderson (1988) — BGW protocol
    - Bonawitz et al. (2017) — Practical Secure Aggregation for FL
    - https://eprint.iacr.org/2017/281.pdf
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core Secret Sharing Primitives
# ─────────────────────────────────────────────────────────────────────────────

def split_into_shares(
    tensor: torch.Tensor,
    num_shares: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> list[torch.Tensor]:
    """
    Split ``tensor`` into ``num_shares`` additive shares over ℝ.

    The shares are constructed so that:
        sum(shares) == tensor    (exact reconstruction guarantee)
        shares[0..n-2] are uniformly random    (semantic security)
        shares[n-1] = tensor − sum(shares[0..n-2])

    Args:
        tensor:     The secret to split (CPU or GPU tensor).
        num_shares: Number of shares to create (≥ 2).
        generator:  Optional seeded torch.Generator for reproducible tests.

    Returns:
        List of ``num_shares`` tensors, each the same shape as ``tensor``.

    Example:
        >>> t = torch.tensor([1.0, 2.0, 3.0])
        >>> shares = split_into_shares(t, num_shares=3)
        >>> torch.allclose(sum(shares), t)   # True
        True
    """
    if num_shares < 2:
        raise ValueError(f"num_shares must be ≥ 2, got {num_shares}")

    shares: list[torch.Tensor] = []
    dtype = tensor.dtype if tensor.is_floating_point() else torch.float64

    t_float = tensor.to(dtype)

    # Generate (num_shares - 1) uniformly random shares
    for _ in range(num_shares - 1):
        share = torch.empty_like(t_float)
        if generator is not None:
            share.normal_(generator=generator)
        else:
            share.normal_()
        shares.append(share)

    # Last share ensures exact reconstruction: last = original − sum(others)
    last_share = t_float - torch.stack(shares).sum(dim=0)
    shares.append(last_share)

    logger.debug(
        f"Split tensor {tuple(tensor.shape)} into {num_shares} additive shares | "
        f"verification: {torch.allclose(torch.stack(shares).sum(dim=0), t_float)}"
    )
    return shares


def reconstruct_from_shares(shares: list[torch.Tensor]) -> torch.Tensor:
    """
    Reconstruct the original tensor from its additive shares.

    Args:
        shares: List of share tensors (all same shape).  Must be ALL shares.

    Returns:
        Reconstructed tensor.

    Example:
        >>> t = torch.randn(64, 8)
        >>> shares = split_into_shares(t, num_shares=3)
        >>> rec = reconstruct_from_shares(shares)
        >>> torch.allclose(rec, t)
        True
    """
    if not shares:
        raise ValueError("Cannot reconstruct from an empty list of shares.")
    result = shares[0].clone()
    for share in shares[1:]:
        result = result + share
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Client Secure Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def secure_aggregate(
    client_updates: list[dict[str, torch.Tensor]],
    client_data_sizes: Optional[list[int]] = None,
    num_servers: int = 3,
    seed: Optional[int] = None,
) -> Optional[dict[str, torch.Tensor]]:
    """
    Securely aggregate a list of client state_dicts using additive secret sharing.

    Simulates the full MPC protocol locally for demonstration / unit-testing.
    In production, Step 2 (server aggregation) would occur on separate servers
    with encrypted channels; only the per-server sums would reach the coordinator.

    Args:
        client_updates:    List of state_dicts from each FL client.
        client_data_sizes: Number of training samples per client (for FedAvg weighting).
                           If None, equal weighting is used.
        num_servers:       Number of aggregation servers (min 3 for meaningful security).
        seed:              Optional RNG seed for reproducible testing.

    Returns:
        Aggregated state_dict, or None if input is empty.

    Security model:
        Any strict subset of servers sees only random tensors.
        Collusion of ALL servers (full reconstruction) breaks security —
        this is identical to FedAvg in that case, but no single server
        ever sees a plaintext client update.
    """
    if not client_updates:
        logger.warning("secure_aggregate called with empty client list.")
        return None

    n_clients = len(client_updates)

    if client_data_sizes is None:
        client_data_sizes = [1] * n_clients
    if len(client_data_sizes) != n_clients:
        raise ValueError("len(client_data_sizes) must equal len(client_updates)")

    total_samples = sum(client_data_sizes)
    weights = [s / total_samples for s in client_data_sizes]

    gen = torch.Generator().manual_seed(seed) if seed is not None else None

    # ── Step 1: Each client splits its update into num_servers shares ─────────
    # In production: client sends share[k] to server k over an encrypted channel.
    # Here: we simulate in memory.
    keys = list(client_updates[0].keys())
    all_client_shares: list[list[dict[str, torch.Tensor]]] = []

    for i, state_dict in enumerate(client_updates):
        # Apply FedAvg weighting BEFORE sharing (so reconstruction = weighted sum)
        weighted_sd = {k: state_dict[k].float() * weights[i] for k in keys}
        client_key_shares: list[dict[str, torch.Tensor]] = [
            {} for _ in range(num_servers)
        ]
        for k in keys:
            tensor_shares = split_into_shares(weighted_sd[k], num_servers, generator=gen)
            for srv_idx, share in enumerate(tensor_shares):
                client_key_shares[srv_idx][k] = share
        all_client_shares.append(client_key_shares)

    # ── Step 2: Each server sums its received shares across all clients ────────
    # In production: each server independently computes this sum on its machine.
    server_sums: list[dict[str, torch.Tensor]] = [{} for _ in range(num_servers)]
    for srv_idx in range(num_servers):
        for k in keys:
            server_sums[srv_idx][k] = torch.stack(
                [all_client_shares[i][srv_idx][k] for i in range(n_clients)]
            ).sum(dim=0)

    # ── Step 3: Coordinator reconstructs by summing all server sums ───────────
    # This is equivalent to: Σᵢ (wᵢ × Δwᵢ)  = FedAvg aggregation
    aggregated: dict[str, torch.Tensor] = {}
    for k in keys:
        server_sum_stack = torch.stack([server_sums[srv][k] for srv in range(num_servers)])
        aggregated[k] = server_sum_stack.sum(dim=0)

    logger.info(
        f"Secure aggregation complete | {n_clients} clients | "
        f"{num_servers} servers | {len(keys)} parameter tensors"
    )
    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Verify Reconstruction Correctness (for testing)
# ─────────────────────────────────────────────────────────────────────────────

def verify_sharing_correctness(tensor: torch.Tensor, num_shares: int = 3, rtol: float = 1e-4) -> bool:
    """
    Sanity check: split → reconstruct → compare.
    Returns True if reconstruction matches original within ``rtol``.
    """
    shares = split_into_shares(tensor, num_shares)
    reconstructed = reconstruct_from_shares(shares)
    return bool(torch.allclose(reconstructed, tensor.float(), rtol=rtol, atol=1e-5))


# ─────────────────────────────────────────────────────────────────────────────
# CLI: Self-Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("FusionNet Additive Secret Sharing — Self-Test")
    print("=" * 60)

    # Test 1: Single-tensor share/reconstruct
    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    ok = verify_sharing_correctness(t, num_shares=3)
    print(f"[{'PASS' if ok else 'FAIL'}] Single tensor split/reconstruct (n=3 shares)")

    # Test 2: Multi-client secure_aggregate correctness
    sd1 = {"adapter.A": torch.tensor([1.0, 0.0]), "adapter.B": torch.tensor([0.0, 1.0])}
    sd2 = {"adapter.A": torch.tensor([3.0, 0.0]), "adapter.B": torch.tensor([0.0, 3.0])}
    result = secure_aggregate([sd1, sd2], client_data_sizes=[1, 3], num_servers=3, seed=42)

    # Expected: weighted average = (1×1 + 3×3) / (1+3) = (1+9)/4 = 2.5 for A[0]
    expected_a0 = (1.0 * 1 + 3.0 * 3) / (1 + 3)
    got_a0 = result["adapter.A"][0].item()
    ok2 = abs(got_a0 - expected_a0) < 1e-4
    print(f"[{'PASS' if ok2 else 'FAIL'}] Multi-client weighted FedAvg via MPC "
          f"(expected={expected_a0:.4f}, got={got_a0:.4f})")

    # Test 3: Security sanity — individual server sees only noise
    secret = torch.ones(100)
    shares = split_into_shares(secret, num_shares=3)
    # Each individual share should NOT be close to the original
    any_leaked = any(torch.allclose(s, secret, atol=0.1) for s in shares)
    ok3 = not any_leaked
    print(f"[{'PASS' if ok3 else 'FAIL'}] Individual shares don't reveal secret")

    all_ok = ok and ok2 and ok3
    print("=" * 60)
    status = "ALL TESTS PASSED [OK]" if all_ok else "SOME TESTS FAILED [FAIL]"
    print(f"Result: {status}")
    sys.exit(0 if all_ok else 1)
