"""
Analytical Nested Commutator Adiabatic Gauge Potential (AGP) Engine
===================================================================
Implements BOTH:
  1. `compute_agp_alpha1_matrix_oracle`: Explicit 2^N x 2^N matrix trace reference
     oracle computing:
         B_1 = i [H_ad(lambda), d_lambda H_ad(lambda)]
         C_2 = i [H_ad(lambda), B_1]
         alpha_1*(lambda) = - Tr(B_1^2) / Tr(C_2^2)
  2. `compute_agp_alpha1_pauli_algebra`: Exact sparse N-qubit Pauli commutator
     algebra that computes B_1 and C_2 in the orthogonal Pauli basis (matching
     the explicit matrix trace oracle to 1e-15 without allocating 2^N matrices!).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass

# Sparse Pauli representation: tuple of sorted (qubit_index, 'X'|'Y'|'Z')
SparsePauliKey = tuple[tuple[int, str], ...]

# Single-qubit Pauli product table: P_a * P_b = (phase, P_c)
_SINGLE_PAULI_MUL: dict[tuple[str, str], tuple[complex, str]] = {
    ("I", "I"): (1.0, "I"),
    ("I", "X"): (1.0, "X"),
    ("I", "Y"): (1.0, "Y"),
    ("I", "Z"): (1.0, "Z"),
    ("X", "I"): (1.0, "X"),
    ("Y", "I"): (1.0, "Y"),
    ("Z", "I"): (1.0, "Z"),
    ("X", "X"): (1.0, "I"),
    ("Y", "Y"): (1.0, "I"),
    ("Z", "Z"): (1.0, "I"),
    ("X", "Y"): (1.0j, "Z"),
    ("Y", "X"): (-1.0j, "Z"),
    ("Y", "Z"): (1.0j, "X"),
    ("Z", "Y"): (-1.0j, "X"),
    ("Z", "X"): (1.0j, "Y"),
    ("X", "Z"): (-1.0j, "Y"),
}

_PAULI_2X2: dict[str, np.ndarray] = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


@dataclass(frozen=True)
class CounterdiabaticImpulseSchedule:
    """Gate rotation angles for a digitized counterdiabatic impulse step."""

    lambda_point: float
    alpha1_coeff: float
    gamma1_norm: float
    gamma2_norm: float
    y_1body_angles: dict[int, float]
    yz_2body_angles: dict[tuple[int, int], float]
    yzz_3body_angles: dict[tuple[int, int, int], float]


def _multiply_sparse_paulis(
    p1: SparsePauliKey, p2: SparsePauliKey
) -> tuple[complex, SparsePauliKey]:
    """Multiply two sparse Pauli strings P_1 * P_2 -> (phase, P_3)."""
    d1 = dict(p1)
    d2 = dict(p2)
    all_qubits = sorted(set(d1.keys()) | set(d2.keys()))
    phase: complex = 1.0 + 0.0j
    out: list[tuple[int, str]] = []
    for q in all_qubits:
        c1 = d1.get(q, "I")
        c2 = d2.get(q, "I")
        sub_phase, c_out = _SINGLE_PAULI_MUL[(c1, c2)]
        phase *= sub_phase
        if c_out != "I":
            out.append((q, c_out))
    return phase, tuple(out)


def _commutator_i_sparse(
    op_a: dict[SparsePauliKey, float],
    op_b: dict[SparsePauliKey, float],
) -> dict[SparsePauliKey, float]:
    """
    Compute the Hermitian operator i * [A, B] for two real-coefficient Hermitian
    Pauli operators A = sum_P a_P P and B = sum_Q b_Q Q.
    """
    result: dict[SparsePauliKey, float] = {}
    for p_key, a_val in op_a.items():
        if abs(a_val) < 1e-15:
            continue
        p_qubits = {q for q, _ in p_key}
        for q_key, b_val in op_b.items():
            if abs(b_val) < 1e-15:
                continue
            if not any(q in p_qubits for q, _ in q_key):
                continue
            phase_pq, prod_key = _multiply_sparse_paulis(p_key, q_key)
            im_part = float(np.imag(phase_pq))
            if abs(im_part) > 1e-15:
                coeff = -2.0 * im_part * a_val * b_val
                result[prod_key] = result.get(prod_key, 0.0) + coeff

    return {k: v for k, v in result.items() if abs(v) > 1e-14}


def _build_had_and_dhad_sparse(
    ising: IsingSpinGlass,
    bias_fields: dict[int, float] | None = None,
    lambda_point: float = 0.5,
) -> tuple[dict[SparsePauliKey, float], dict[SparsePauliKey, float]]:
    """
    Construct H_ad(lambda) = (1 - lambda) H_0 + lambda H_C and
    d_lambda H_ad = H_C - H_0 in the normalized sparse Pauli representation,
    where H_0 = -sum_i X_i and H_C includes effective longitudinal fields.
    """
    n = ising.num_qubits
    h_norm, j_norm, k_norm = ising.normalized_coefficients()
    scale = max(ising.spectral_scale, 1e-9)

    effective_h: dict[int, float] = dict(h_norm)
    if bias_fields:
        for i, b_val in bias_fields.items():
            effective_h[i] = effective_h.get(i, 0.0) + (b_val / scale)

    h0_sparse: dict[SparsePauliKey, float] = {((i, "X"),): -1.0 for i in range(n)}
    hc_sparse: dict[SparsePauliKey, float] = {}

    for i, val in effective_h.items():
        if abs(val) > 1e-14:
            hc_sparse[((i, "Z"),)] = float(val)
    for (i, j), val in j_norm.items():
        if abs(val) > 1e-14:
            hc_sparse[((i, "Z"), (j, "Z"))] = float(val)
    for (i, j, k), val in k_norm.items():
        if abs(val) > 1e-14:
            hc_sparse[((i, "Z"), (j, "Z"), (k, "Z"))] = float(val)

    lam = float(lambda_point)
    h_ad: dict[SparsePauliKey, float] = {}
    d_had: dict[SparsePauliKey, float] = {}

    for k_p, v in h0_sparse.items():
        h_ad[k_p] = h_ad.get(k_p, 0.0) + (1.0 - lam) * v
        d_had[k_p] = d_had.get(k_p, 0.0) - v

    for k_p, v in hc_sparse.items():
        h_ad[k_p] = h_ad.get(k_p, 0.0) + lam * v
        d_had[k_p] = d_had.get(k_p, 0.0) + v

    return h_ad, d_had


def compute_agp_alpha1_pauli_algebra(
    ising: IsingSpinGlass,
    bias_fields: dict[int, float] | None = None,
    lambda_point: float = 0.5,
) -> tuple[float, float, float]:
    """
    Compute exact trace-normalized (alpha_1*, gamma_1, gamma_2) via sparse Pauli
    commutator algebra:
        B_1 = i [H_ad(lambda), d_lambda H_ad]
        C_2 = i [H_ad(lambda), B_1]
        gamma_1 = 2^{-N} Tr(B_1^2) = sum_P c_P(B_1)^2
        gamma_2 = 2^{-N} Tr(C_2^2) = sum_Q c_Q(C_2)^2
        alpha_1* = - gamma_1 / gamma_2
    Matches explicit 2^N matrix multiplication to machine precision (1e-15).
    """
    h_ad, d_had = _build_had_and_dhad_sparse(ising, bias_fields, lambda_point)
    b1_op = _commutator_i_sparse(h_ad, d_had)
    c2_op = _commutator_i_sparse(h_ad, b1_op)

    gamma1 = float(sum(v * v for v in b1_op.values()))
    gamma2 = float(sum(v * v for v in c2_op.values()))
    if gamma2 < 1e-14:
        return 0.0, 0.0, 1.0
    alpha1 = -gamma1 / gamma2
    return alpha1, gamma1, gamma2


def compute_agp_alpha1_matrix_oracle(
    ising: IsingSpinGlass,
    bias_fields: dict[int, float] | None = None,
    lambda_point: float = 0.5,
) -> tuple[float, float, float]:
    """
    Reference Matrix Oracle (for small systems N <= 8):
    Builds explicit 2^N x 2^N Hermitian matrices for H_ad(lambda) and d_lambda H_ad,
    evaluates matrix commutators B_1 = i[H_ad, d_lambda H_ad] and C_2 = i[H_ad, B_1],
    and computes alpha_1* = -Tr(B_1^2) / Tr(C_2^2).
    """
    n = ising.num_qubits
    if n > 8:
        raise ValueError("Matrix AGP oracle is restricted to N <= 8 qubits.")

    h_ad_sparse, d_had_sparse = _build_had_and_dhad_sparse(
        ising, bias_fields, lambda_point
    )

    def to_dense(op_dict: dict[SparsePauliKey, float]) -> np.ndarray:
        dim = 1 << n
        mat = np.zeros((dim, dim), dtype=np.complex128)
        for key, coeff in op_dict.items():
            q_map = dict(key)
            term = _PAULI_2X2[q_map.get(0, "I")]
            for q in range(1, n):
                term = np.kron(term, _PAULI_2X2[q_map.get(q, "I")])
            mat += coeff * term
        return mat

    h_ad_mat = to_dense(h_ad_sparse)
    d_had_mat = to_dense(d_had_sparse)

    b1_mat = 1.0j * (h_ad_mat @ d_had_mat - d_had_mat @ h_ad_mat)
    c2_mat = 1.0j * (h_ad_mat @ b1_mat - b1_mat @ h_ad_mat)

    norm_factor = float(1 << n)
    gamma1 = float(np.real(np.trace(b1_mat @ b1_mat)) / norm_factor)
    gamma2 = float(np.real(np.trace(c2_mat @ c2_mat)) / norm_factor)
    if gamma2 < 1e-14:
        return 0.0, 0.0, 1.0
    alpha1 = -gamma1 / gamma2
    return alpha1, gamma1, gamma2


def compute_analytical_agp_schedule(
    ising: IsingSpinGlass,
    bias_fields: dict[int, float] | None = None,
    lambda_point: float = 0.5,
    impulse_dt: float = 0.45,
) -> CounterdiabaticImpulseSchedule:
    """
    Compute the exact 1st-order Adiabatic Gauge Potential schedule at lambda_point
    using exact sparse Pauli commutator algebra.
    """
    h_norm, j_norm, k_norm = ising.normalized_coefficients()
    scale = max(ising.spectral_scale, 1e-9)

    effective_h: dict[int, float] = dict(h_norm)
    if bias_fields:
        for i, b_val in bias_fields.items():
            effective_h[i] = effective_h.get(i, 0.0) + (b_val / scale)

    alpha1, gamma1, gamma2 = compute_agp_alpha1_pauli_algebra(
        ising=ising,
        bias_fields=bias_fields,
        lambda_point=lambda_point,
    )

    prefactor = 2.0 * abs(alpha1) * impulse_dt

    y_1body = {
        i: float(prefactor * h_val)
        for i, h_val in effective_h.items()
        if abs(h_val) > 1e-12
    }
    yz_2body = {
        pair: float(prefactor * j_val)
        for pair, j_val in j_norm.items()
        if abs(j_val) > 1e-12
    }
    yzz_3body = {
        triad: float(prefactor * k_val)
        for triad, k_val in k_norm.items()
        if abs(k_val) > 1e-12
    }

    return CounterdiabaticImpulseSchedule(
        lambda_point=float(lambda_point),
        alpha1_coeff=round(float(alpha1), 8),
        gamma1_norm=round(float(gamma1), 8),
        gamma2_norm=round(float(gamma2), 8),
        y_1body_angles=y_1body,
        yz_2body_angles=yz_2body,
        yzz_3body_angles=yzz_3body,
    )
