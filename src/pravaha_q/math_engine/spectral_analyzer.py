"""
Spectral Gap, Degeneracy & Spin-Glass Frustration Analyzer
==========================================================
Computes exact ground state energy E_0, first excited state energy E_1,
spectral gap Delta_E = E_1 - E_0, and geometric frustration index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass


@dataclass(frozen=True)
class SpectralReport:
    """Complete spectral diagnosis of an Ising Spin-Glass Hamiltonian."""

    num_qubits: int
    ground_energy: float
    first_excited_energy: float
    max_energy: float
    spectral_gap: float
    ground_state_bitstring: str
    ground_state_degeneracy: int
    frustration_index: float


def analyze_ising_spectrum(ising: IsingSpinGlass) -> SpectralReport:
    """
    Compute the exact eigenspectrum report for an IsingSpinGlass (for N <= 20 qubits).
    """
    n = ising.num_qubits
    if n > 20:
        raise ValueError(
            f"Exact 2^N spectral diagonalization requires N <= 20 (got N={n}). "
            "Use MPS Tensor Network or PCE for larger systems."
        )

    diag = ising.diagonal_vector()
    min_idx = int(np.argmin(diag))
    e_min = float(diag[min_idx])
    e_max = float(np.max(diag))

    ground_mask = np.isclose(diag, e_min, atol=1e-7)
    degeneracy = int(np.sum(ground_mask))

    excited_energies = diag[~ground_mask]
    e_first = float(np.min(excited_energies)) if excited_energies.size > 0 else e_min
    spectral_gap = float(max(e_first - e_min, 0.0))

    best_bitstring = format(min_idx, f"0{n}b")
    best_spins = [1 - 2 * int(c) for c in best_bitstring]

    # Compute spin-glass frustration fraction: ratio of unsatisfied coupling energy
    # in the true ground state vs total absolute coupling weight
    total_coupling_mag = 0.0
    unsatisfied_mag = 0.0
    for (i, j), j_val in ising.j_quadratic.items():
        mag = abs(j_val)
        total_coupling_mag += mag
        term_e = j_val * best_spins[i] * best_spins[j]
        if term_e > 0:
            unsatisfied_mag += mag

    for (i, j, k), k_val in ising.k_cubic.items():
        mag = abs(k_val)
        total_coupling_mag += mag
        term_e = k_val * best_spins[i] * best_spins[j] * best_spins[k]
        if term_e > 0:
            unsatisfied_mag += mag

    frustration = (
        float(unsatisfied_mag / total_coupling_mag) if total_coupling_mag > 1e-12 else 0.0
    )

    return SpectralReport(
        num_qubits=n,
        ground_energy=e_min,
        first_excited_energy=e_first,
        max_energy=e_max,
        spectral_gap=spectral_gap,
        ground_state_bitstring=best_bitstring,
        ground_state_degeneracy=degeneracy,
        frustration_index=round(frustration, 4),
    )
