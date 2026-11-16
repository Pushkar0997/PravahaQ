"""
Analytical Nested Commutator Adiabatic Gauge Potential (AGP) Engine
===================================================================
Computes the exact closed-form variational counterdiabatic gauge coefficients
alpha_1*(lambda) for a 3-local Ising Spin-Glass Hamiltonian without requiring
a classical outer optimization loop:

    A_lambda^{(1)} = i * alpha_1(lambda) * [H_ad(lambda), d_lambda H_ad(lambda)]
                   = 2 * alpha_1(lambda) * (
                         sum_i h_i Y_i
                       + sum_{i<j} J_{ij} (Y_i Z_j + Z_i Y_j)
                       + sum_{i<j<k} K_{ijk} (Y_i Z_j Z_k + Z_i Y_j Z_k + Z_i Z_j Y_k)
                     )
"""

from __future__ import annotations

from dataclasses import dataclass

from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass


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


def compute_analytical_agp_schedule(
    ising: IsingSpinGlass,
    bias_fields: dict[int, float] | None = None,
    lambda_point: float = 0.5,
    impulse_dt: float = 0.45,
) -> CounterdiabaticImpulseSchedule:
    """
    Compute the exact analytical 1st-order Claeys-Berry-Sels Adiabatic Gauge
    Potential schedule at schedule parameter lambda_point in (0, 1).
    """
    h_norm, j_norm, k_norm = ising.normalized_coefficients()
    scale = max(ising.spectral_scale, 1e-9)

    # Blend normalized intrinsic fields with iterative longitudinal bias fields
    effective_h: dict[int, float] = dict(h_norm)
    if bias_fields:
        for i, b_val in bias_fields.items():
            effective_h[i] = effective_h.get(i, 0.0) + (b_val / scale)

    sum_h2 = sum(v * v for v in effective_h.values())
    sum_h4 = sum(v**4 for v in effective_h.values())
    sum_j2 = sum(v * v for v in j_norm.values())
    sum_j4 = sum(v**4 for v in j_norm.values())
    sum_k2 = sum(v * v for v in k_norm.values())
    sum_k4 = sum(v**4 for v in k_norm.values())

    # Exact Hilbert-Schmidt action numerator Gamma_1 and denominator Gamma_2
    gamma1 = sum_h2 + 2.0 * sum_j2 + 3.0 * sum_k2
    lam = float(lambda_point)
    one_m_lam = 1.0 - lam

    term_mixer = 4.0 * (one_m_lam**2) * (sum_h2 + 4.0 * sum_j2 + 9.0 * sum_k2)
    term_cost = 4.0 * (lam**2) * (
        sum_h4 + 6.0 * sum_j4 + 12.0 * sum_k4 + 3.0 * sum_h2 * sum_j2
    )
    gamma2 = max(term_mixer + term_cost, 1e-6)

    alpha1 = -gamma1 / gamma2
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
        lambda_point=lam,
        alpha1_coeff=round(float(alpha1), 6),
        gamma1_norm=round(float(gamma1), 6),
        gamma2_norm=round(float(gamma2), 6),
        y_1body_angles=y_1body,
        yz_2body_angles=yz_2body,
        yzz_3body_angles=yzz_3body,
    )
