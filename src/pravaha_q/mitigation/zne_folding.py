"""
Genuine Unitary-Folded Zero-Noise Extrapolation (ZNE) with Uncertainty Propagation
==================================================================================
Executes actual folded `QuantumCircuitIR` instances U_c = U (U^dagger U)^{(c-1)/2}
at odd scale factors c in {1, 3, 5} under gate-dependent depolarizing noise,
computes Lagrange/Richardson zero-noise extrapolation, and propagates finite-shot
standard errors:
    sigma_ZNE = sqrt( sum_m gamma_m^2 * sigma_m^2 ).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pravaha_q.algorithms.bf_dcqo import sample_bitstring_counts
from pravaha_q.compiler.circuit_ir import QuantumCircuitIR
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass


@dataclass(frozen=True)
class ZNEExtrapolationReport:
    """Complete Zero-Noise Extrapolation report with statistical uncertainty."""

    scale_factors: tuple[float, ...]
    folded_2q_gate_counts: tuple[int, ...]
    measured_expectations: tuple[float, ...]
    measured_std_errors: tuple[float, ...]
    richardson_weights: tuple[float, ...]
    extrapolated_energy: float
    extrapolated_std_error: float


def compute_richardson_weights(
    scale_factors: tuple[float, ...] = (1.0, 3.0, 5.0),
) -> np.ndarray:
    """Compute Lagrange polynomial extrapolation weights gamma_m at c = 0."""
    c = np.array(scale_factors, dtype=np.float64)
    m = len(c)
    weights = np.empty(m, dtype=np.float64)
    for i in range(m):
        prod = 1.0
        for j in range(m):
            if i != j:
                prod *= c[j] / (c[j] - c[i])
        weights[i] = prod
    return weights


def richardson_zero_noise_extrapolation(
    scale_factors: tuple[float, ...] = (1.0, 3.0, 5.0),
    measured_expectations: tuple[float, ...] = (10.0, 12.5, 15.2),
) -> float:
    """Compute Richardson polynomial extrapolation estimate at scale factor c = 0."""
    weights = compute_richardson_weights(scale_factors)
    y = np.array(measured_expectations, dtype=np.float64)
    return float(np.sum(weights * y))


def execute_circuit_zne(
    circuit: QuantumCircuitIR,
    ising: IsingSpinGlass,
    scale_factors: tuple[int, ...] = (1, 3, 5),
    per_gate_depol_error: float = 0.008,
    shots: int = 2048,
    seed: int = 42,
) -> ZNEExtrapolationReport:
    """
    Construct real unitary-folded circuits U_c = circuit.fold_unitary(c) for each
    scale factor c in `scale_factors`, simulate each under gate-count-proportional
    depolarizing noise, sample `shots` measurements, and compute the Richardson
    zero-noise estimate and its propagated standard error.
    """
    n = circuit.num_qubits
    weights = compute_richardson_weights(tuple(float(c) for c in scale_factors))

    gate_counts_2q: list[int] = []
    expectations: list[float] = []
    std_errors: list[float] = []

    for idx, c in enumerate(scale_factors):
        folded_ir = circuit.fold_unitary(int(c))
        gate_counts_2q.append(folded_ir.count_2q_gates())

        state_c = folded_ir.simulate_statevector(
            per_gate_depol_error=per_gate_depol_error
        )
        probs_c = np.abs(state_c) ** 2
        probs_c = probs_c / np.sum(probs_c)

        counts_c = sample_bitstring_counts(
            probs_c, num_qubits=n, shots=shots, seed=seed + idx
        )
        energies = np.array(
            [ising.evaluate_bitstring(bstr) for bstr in counts_c.keys()],
            dtype=np.float64,
        )
        weights_s = np.array(list(counts_c.values()), dtype=np.float64)
        total_s = float(np.sum(weights_s))

        mean_e = float(np.sum(energies * weights_s) / total_s)
        var_e = float(np.sum(weights_s * ((energies - mean_e) ** 2)) / total_s)
        se_e = float(np.sqrt(max(var_e, 0.0) / max(total_s, 1.0)))

        expectations.append(round(mean_e, 6))
        std_errors.append(round(se_e, 6))

    y_arr = np.array(expectations, dtype=np.float64)
    se_arr = np.array(std_errors, dtype=np.float64)

    zne_val = float(np.sum(weights * y_arr))
    zne_se = float(np.sqrt(np.sum((weights**2) * (se_arr**2))))

    return ZNEExtrapolationReport(
        scale_factors=tuple(float(c) for c in scale_factors),
        folded_2q_gate_counts=tuple(gate_counts_2q),
        measured_expectations=tuple(expectations),
        measured_std_errors=tuple(std_errors),
        richardson_weights=tuple(round(float(w), 5) for w in weights),
        extrapolated_energy=round(zne_val, 6),
        extrapolated_std_error=round(zne_se, 6),
    )
