"""
Goemans-Williamson SDP Relaxation, Rigorous Lower Bound & Hyperplane Rounding
=============================================================================
Computes a provably valid lower bound L <= E* for any 2-body or 3-body Ising
Hamiltonian:
  - 1-body & 2-body terms are bounded via the homogenized (N+1)x(N+1) Rayleigh
    minimum eigenvalue (n + 1) * lambda_min(M).
  - 3-body cubic terms are bounded via the exact L1 dual bound -sum_{i<j<k} |K_{ijk}|.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass


@dataclass(frozen=True)
class SDPRelaxationResult:
    """Result of spectral SDP relaxation, hyperplane rounding, and rigorous lower bound."""

    continuous_probabilities: list[float]
    best_bitstring: str
    best_energy: float
    sdp_lower_bound: float


def solve_gw_sdp_relaxation(
    ising: IsingSpinGlass,
    num_hyperplanes: int = 64,
    seed: int = 42,
) -> SDPRelaxationResult:
    """
    Solve the homogenized (N+1)-dimensional spectral SDP relaxation of the Ising
    Hamiltonian and compute a guaranteed valid lower bound L <= E*.
    """
    n = ising.num_qubits
    mat = np.zeros((n + 1, n + 1), dtype=np.float64)
    for i, h_val in ising.h_linear.items():
        mat[0, i + 1] = 0.5 * h_val
        mat[i + 1, 0] = 0.5 * h_val
    for (i, j), j_val in ising.j_quadratic.items():
        mat[i + 1, j + 1] = 0.5 * j_val
        mat[j + 1, i + 1] = 0.5 * j_val

    eigvals, eigvecs = np.linalg.eigh(mat)
    rank_k = min(n + 1, 4)
    weights = np.exp(-2.0 * (eigvals[:rank_k] - eigvals[0]))
    embedding = eigvecs[:, :rank_k] * np.sqrt(weights)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    embedding = embedding / norms

    # Rigorous lower bound including 3-body cubic dual bound: L <= E*
    cubic_l1_bound = sum(abs(v) for v in ising.k_cubic.values())
    sdp_lb = float(ising.constant_offset + (n + 1) * eigvals[0] - cubic_l1_bound)

    rng = np.random.default_rng(seed)
    best_energy = float("inf")
    best_bits = [0] * n
    bit_accum = np.zeros(n, dtype=np.float64)

    for _ in range(num_hyperplanes):
        r_vec = rng.normal(size=rank_k)
        projs = embedding @ r_vec
        ref_sign = 1 if projs[0] >= 0.0 else -1
        spins = [
            1 if (projs[i + 1] * ref_sign) >= 0.0 else -1 for i in range(n)
        ]
        bits = [(1 - s) // 2 for s in spins]
        e = ising.evaluate_spin_energy(spins)

        bit_accum += np.array(bits, dtype=np.float64)
        if e < best_energy:
            best_energy = e
            best_bits = bits

    probs = np.clip(bit_accum / max(num_hyperplanes, 1), 0.05, 0.95)
    blended_c = [
        round(float(0.65 * best_bits[i] + 0.35 * probs[i]), 5) for i in range(n)
    ]

    return SDPRelaxationResult(
        continuous_probabilities=blended_c,
        best_bitstring="".join(str(b) for b in best_bits),
        best_energy=round(best_energy, 6),
        sdp_lower_bound=round(sdp_lb, 6),
    )
