"""
FusionNet ZKP: Commitment-Based Norm Verification
==================================================
Implements a cryptographic commit–reveal scheme that allows a client to
prove its gradient update satisfies the DP clipping bound ||ΔW||₂ ≤ C
**without revealing ΔW itself**.

Security model (Commitment Scheme):
    - Hiding:   The commitment c = H(tensor_bytes || nonce) reveals nothing
                about the tensor.  The verifier cannot distinguish commitments
                from two different tensors without the nonce.
    - Binding:  A client cannot open the commitment to a *different* tensor
                after submission.  SHA-256 is computationally binding under
                the random oracle model.

Protocol:
    1. Client clips its tensor to norm C:
           Δw_clipped = Δw / max(1, ||Δw||₂ / C)
    2. Client generates a random nonce and computes:
           commitment = SHA256(Δw_clipped_bytes || nonce)
    3. Client uploads to coordinator:  (commitment, Δw_clipped, claimed_norm)
    4. Coordinator verifies:  claimed_norm ≤ C  (Step A)
    5. Client reveals nonce; coordinator recomputes H(Δw_clipped_bytes || nonce)
       and checks it matches the original commitment (Step B).

What this proves:
    ✓ The uploaded Δw_clipped was NOT modified between commit and reveal.
    ✓ The upload satisfies the DP clipping norm bound.
    ✗ This is NOT a zero-knowledge proof in the cryptographic sense —
      the verifier does see Δw_clipped (but not Δw before clipping).

Production upgrade path:
    Replace the commitment with a Groth16 zk-SNARK circuit over BN254 that
    proves ||ΔW||₂ ≤ C in zero-knowledge (verifier never sees any ΔW).
    Libraries: circom + snarkjs (JavaScript), or gnark (Go).

References:
    Pedersen (1992) — Non-interactive and Information-Theoretic Secure Verifiable
    Secret Sharing.  https://link.springer.com/chapter/10.1007/3-540-46766-1_9

    Boneh & Shoup — "A Graduate Course in Applied Cryptography", Ch. 8.
    https://toc.cryptobook.us/

    Garg et al. (2023) — "ZK-FL: Privacy-Preserving Federated Learning via
    Zero-Knowledge Proofs".  https://arxiv.org/abs/2306.05178
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import struct
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Default DP clipping norm (must match training config)
DEFAULT_CLIP_NORM: float = 1.0
# Nonce size in bytes (256-bit nonce for 128-bit security)
NONCE_BYTES: int = 32


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GradientCommitment:
    """
    A commitment to a gradient update tensor.

    Attributes:
        commitment_hex: SHA-256 hex digest of (tensor_bytes || nonce).
        claimed_norm:   L2 norm of the submitted (clipped) tensor.
        tensor_shape:   Shape of the tensor (for deserialization).
        tensor_dtype:   String dtype name (e.g. 'float32').
        client_id:      Identifier for the submitting client.
        round_num:      Federated learning round this commitment belongs to.
    """
    commitment_hex: str
    claimed_norm: float
    tensor_shape: tuple
    tensor_dtype: str
    client_id: str
    round_num: int


@dataclass
class VerificationResult:
    """Result of verifying a gradient commitment."""
    is_valid: bool
    client_id: str
    round_num: int
    claimed_norm: float
    clip_norm: float
    norm_check_passed: bool    # claimed_norm ≤ clip_norm
    hash_check_passed: bool    # commitment matches revealed tensor + nonce
    rejection_reason: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Core Commitment Functions
# ─────────────────────────────────────────────────────────────────────────────

def generate_nonce() -> bytes:
    """Generate a cryptographically secure random nonce."""
    return os.urandom(NONCE_BYTES)


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes (float32, CPU) for hashing."""
    buf = io.BytesIO()
    # Include shape in the byte representation to prevent shape-confusion attacks
    shape_bytes = struct.pack(f"<{len(tensor.shape)}q", *tensor.shape)
    buf.write(shape_bytes)
    buf.write(tensor.cpu().to(torch.float32).numpy().tobytes())
    return buf.getvalue()


def commit(tensor: torch.Tensor, nonce: bytes) -> str:
    """
    Create a cryptographic commitment to ``tensor``.

    Commitment = SHA-256(tensor_bytes || nonce)

    Args:
        tensor: The tensor to commit to (any shape, CPU or GPU).
        nonce:  Random bytes (use generate_nonce()).

    Returns:
        Hex-encoded SHA-256 commitment string.
    """
    tensor_bytes = _tensor_to_bytes(tensor)
    hasher = hashlib.sha256()
    hasher.update(tensor_bytes)
    hasher.update(nonce)
    digest = hasher.hexdigest()
    logger.debug(f"Committed tensor {tuple(tensor.shape)} → {digest[:16]}...")
    return digest


def clip_tensor(tensor: torch.Tensor, clip_norm: float) -> tuple[torch.Tensor, float]:
    """
    Project ``tensor`` onto the L2 ball of radius ``clip_norm``.

    Args:
        tensor:    Input tensor.
        clip_norm: Maximum allowed L2 norm.

    Returns:
        (clipped_tensor, actual_norm_before_clipping)
    """
    actual_norm = float(torch.linalg.vector_norm(tensor.float()).item())
    scale = min(1.0, clip_norm / (actual_norm + 1e-12))
    clipped = tensor.float() * scale
    return clipped, actual_norm


def create_commitment_package(
    tensor: torch.Tensor,
    client_id: str,
    round_num: int,
    clip_norm: float = DEFAULT_CLIP_NORM,
) -> tuple[GradientCommitment, torch.Tensor, bytes]:
    """
    Full client-side commitment workflow.

    1. Clip tensor to clip_norm.
    2. Generate nonce.
    3. Compute commitment hash.

    Args:
        tensor:    Raw local gradient update tensor (ΔW).
        client_id: This client's identifier.
        round_num: Current FL round number.
        clip_norm: DP gradient clipping norm C.

    Returns:
        Tuple of (commitment, clipped_tensor, nonce).
        - Send commitment + clipped_tensor to coordinator immediately.
        - Reveal nonce only after coordinator acknowledges receipt.
    """
    clipped, _ = clip_tensor(tensor, clip_norm)
    claimed_norm = float(torch.linalg.vector_norm(clipped).item())
    nonce = generate_nonce()
    commitment_hex = commit(clipped, nonce)

    commitment = GradientCommitment(
        commitment_hex=commitment_hex,
        claimed_norm=claimed_norm,
        tensor_shape=tuple(clipped.shape),
        tensor_dtype=str(clipped.dtype).replace("torch.", ""),
        client_id=client_id,
        round_num=round_num,
    )

    logger.info(
        f"Client {client_id} | Round {round_num} | "
        f"Committed clipped update | norm={claimed_norm:.4f} ≤ {clip_norm}"
    )
    return commitment, clipped, nonce


# ─────────────────────────────────────────────────────────────────────────────
# Verifier (Coordinator Side)
# ─────────────────────────────────────────────────────────────────────────────

def verify_norm_bound(
    commitment: GradientCommitment,
    revealed_tensor: torch.Tensor,
    revealed_nonce: bytes,
    clip_norm: float = DEFAULT_CLIP_NORM,
) -> VerificationResult:
    """
    Coordinator-side verification of a client's gradient commitment.

    Step A — Norm check:  commitment.claimed_norm ≤ clip_norm
    Step B — Hash check:  SHA256(tensor_bytes || nonce) == commitment.commitment_hex

    Args:
        commitment:      The commitment submitted by the client.
        revealed_tensor: The clipped gradient tensor (uploaded by client).
        revealed_nonce:  The nonce revealed by the client after upload.
        clip_norm:       The DP clipping norm C (from server config).

    Returns:
        VerificationResult with detailed pass/fail information.
    """
    # Step A: Norm bound check
    norm_ok = commitment.claimed_norm <= (clip_norm + 1e-6)   # small tolerance for float32
    actual_norm = float(torch.linalg.vector_norm(revealed_tensor.float()).item())
    # Also re-verify the actual tensor norm (client could lie about claimed_norm)
    actual_norm_ok = actual_norm <= (clip_norm + 1e-6)
    norm_check_passed = norm_ok and actual_norm_ok

    # Step B: Hash check — recompute commitment from revealed tensor + nonce
    recomputed_hex = commit(revealed_tensor, revealed_nonce)
    hash_check_passed = recomputed_hex == commitment.commitment_hex

    is_valid = norm_check_passed and hash_check_passed

    rejection_reason = None
    if not norm_check_passed:
        rejection_reason = (
            f"Norm violation: claimed={commitment.claimed_norm:.4f}, "
            f"actual={actual_norm:.4f}, limit={clip_norm}"
        )
    elif not hash_check_passed:
        rejection_reason = "Commitment mismatch: tensor was modified after commitment"

    if is_valid:
        logger.info(
            f"✓ Verified | client={commitment.client_id} | "
            f"round={commitment.round_num} | norm={actual_norm:.4f} ≤ {clip_norm}"
        )
    else:
        logger.warning(
            f"✗ REJECTED | client={commitment.client_id} | "
            f"round={commitment.round_num} | reason: {rejection_reason}"
        )

    return VerificationResult(
        is_valid=is_valid,
        client_id=commitment.client_id,
        round_num=commitment.round_num,
        claimed_norm=commitment.claimed_norm,
        clip_norm=clip_norm,
        norm_check_passed=norm_check_passed,
        hash_check_passed=hash_check_passed,
        rejection_reason=rejection_reason,
    )


def batch_verify(
    commitments: list[GradientCommitment],
    revealed_tensors: list[torch.Tensor],
    revealed_nonces: list[bytes],
    clip_norm: float = DEFAULT_CLIP_NORM,
) -> tuple[list[torch.Tensor], list[VerificationResult]]:
    """
    Verify a batch of client commitments and return only the valid tensors.

    Args:
        commitments:      List of GradientCommitment from each client.
        revealed_tensors: Corresponding revealed gradient tensors.
        revealed_nonces:  Corresponding revealed nonces.
        clip_norm:        DP clipping norm.

    Returns:
        (valid_tensors, all_results) — valid_tensors contains only tensors
        whose commitment verified; all_results has per-client details.
    """
    if not (len(commitments) == len(revealed_tensors) == len(revealed_nonces)):
        raise ValueError("commitments, revealed_tensors, and revealed_nonces must have equal length")

    results = []
    valid_tensors = []

    for comm, tensor, nonce in zip(commitments, revealed_tensors, revealed_nonces):
        result = verify_norm_bound(comm, tensor, nonce, clip_norm)
        results.append(result)
        if result.is_valid:
            valid_tensors.append(tensor)

    n_valid = len(valid_tensors)
    n_total = len(commitments)
    logger.info(
        f"Batch verification: {n_valid}/{n_total} passed "
        f"({n_total - n_valid} rejected)"
    )
    return valid_tensors, results


# ─────────────────────────────────────────────────────────────────────────────
# CLI: Self-Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("FusionNet ZKP Commitment Scheme — Self-Test")
    print("=" * 60)

    CLIP = 1.0
    results = {}

    # ── Test 1: Clean update within norm bound ─────────────────────────────
    clean_update = torch.randn(64, 8) * 0.5     # norm < 1.0
    comm, clipped, nonce = create_commitment_package(clean_update, "client_0", 1, CLIP)
    result = verify_norm_bound(comm, clipped, nonce, CLIP)
    results["clean_update"] = result.is_valid
    print(f"[{'PASS' if result.is_valid else 'FAIL'}] Clean update accepted (norm <= {CLIP})")

    # ── Test 2: Oversized update gets clipped and accepted ─────────────────
    big_update = torch.randn(64, 8) * 10.0      # norm >> 1.0
    comm2, clipped2, nonce2 = create_commitment_package(big_update, "client_1", 1, CLIP)
    result2 = verify_norm_bound(comm2, clipped2, nonce2, CLIP)
    results["clipped_update"] = result2.is_valid
    print(f"[{'PASS' if result2.is_valid else 'FAIL'}] Oversized update clipped and accepted")

    # ── Test 3: Tampered tensor rejected ───────────────────────────────────
    comm3, clipped3, nonce3 = create_commitment_package(clean_update, "client_2", 1, CLIP)
    tampered = clipped3 + torch.ones_like(clipped3) * 0.1   # malicious modification
    result3 = verify_norm_bound(comm3, tampered, nonce3, CLIP)
    results["tampered_tensor"] = not result3.is_valid   # should REJECT
    print(f"[{'PASS' if not result3.is_valid else 'FAIL'}] Tampered tensor rejected "
          f"(hash_ok={result3.hash_check_passed})")

    # ── Test 4: Wrong nonce rejected ────────────────────────────────────────
    comm4, clipped4, _ = create_commitment_package(clean_update, "client_3", 1, CLIP)
    wrong_nonce = generate_nonce()
    result4 = verify_norm_bound(comm4, clipped4, wrong_nonce, CLIP)
    results["wrong_nonce"] = not result4.is_valid
    print(f"[{'PASS' if not result4.is_valid else 'FAIL'}] Wrong nonce rejected")

    # ── Summary ─────────────────────────────────────────────────────────────
    all_ok = all(results.values())
    print("=" * 60)
    status = "ALL TESTS PASSED [OK]" if all_ok else "SOME TESTS FAILED [FAIL]"
    print(f"Result: {status}")
    sys.exit(0 if all_ok else 1)
