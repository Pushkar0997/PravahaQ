"""2D Variational (gamma, beta) Energy Landscape & Bias-Field Trajectory Scanner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pravaha_q.algorithms.bf_dcqo import apply_single_qubit_rx
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass


@dataclass(frozen=True)
class LandscapeSurface2D:
    """2D grid of expectation values <psi(gamma, beta)| H_C |psi(gamma, beta)>."""

    gamma_grid: list[float]
    beta_grid: list[float]
    energy_matrix: list[list[float]]
    min_energy_point: tuple[float, float, float]


def scan_energy_landscape_2d(
    ising: IsingSpinGlass,
    resolution: int = 12,
    gamma_bounds: tuple[float, float] = (0.05, 1.50),
    beta_bounds: tuple[float, float] = (0.05, 1.50),
) -> LandscapeSurface2D:
    """Scan the 2D (gamma, beta) expectation surface for a 3-local IsingSpinGlass."""
    n = ising.num_qubits
    dim = 1 << n
    diag_cost = ising.diagonal_vector()
    scale = max(ising.spectral_scale, 1e-9)
    norm_diag = (diag_cost - ising.constant_offset) / scale

    gammas = np.linspace(gamma_bounds[0], gamma_bounds[1], resolution)
    betas = np.linspace(beta_bounds[0], beta_bounds[1], resolution)
    surface: list[list[float]] = []

    plus_state = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    min_val = float("inf")
    min_pt = (float(gammas[0]), float(betas[0]), 0.0)

    for g_val in gammas:
        row: list[float] = []
        phased = plus_state * np.exp(-1.0j * float(g_val) * norm_diag)
        for b_val in betas:
            psi = phased.reshape((2,) * n)
            for q in range(n):
                psi = apply_single_qubit_rx(psi, q, -2.0 * float(b_val))
            probs = np.abs(psi.reshape(-1)) ** 2
            exp_e = float(np.sum(probs * diag_cost))
            row.append(round(exp_e, 5))
            if exp_e < min_val:
                min_val = exp_e
                min_pt = (
                    round(float(g_val), 4),
                    round(float(b_val), 4),
                    round(exp_e, 5),
                )
        surface.append(row)

    return LandscapeSurface2D(
        gamma_grid=[round(float(g), 4) for g in gammas],
        beta_grid=[round(float(b), 4) for b in betas],
        energy_matrix=surface,
        min_energy_point=min_pt,
    )
