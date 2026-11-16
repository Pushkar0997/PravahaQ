"""
Conditional Value-at-Risk (CVaR_alpha) Tail Expectation Estimator
==================================================================
Filters noisy hardware shot samples, retaining only the lowest alpha-fraction
(best energy tail) to improve optimization gradient convergence.

Reference:
  Barkoutsos, P. K., et al. (2020). "Improving Variational Quantum Optimization
  using CVaR." Quantum, 4, 256.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def compute_cvar_expectation(
    bitstring_counts: dict[str, int],
    energy_evaluator_fn: Callable[[str], float],
    alpha: float = 0.20,
) -> float:
    """
    Compute Conditional Value-at-Risk expectation value CVaR_alpha over sampled bitstrings.
    """
    alpha = float(np.clip(alpha, 0.01, 1.0))
    records: list[tuple[float, int]] = []
    for bstr, count in bitstring_counts.items():
        e = float(energy_evaluator_fn(bstr))
        records.append((e, count))

    # Sort by non-decreasing energy (best energies first)
    records.sort(key=lambda item: item[0])
    total_shots = sum(c for _, c in records)
    cutoff_shots = max(int(np.ceil(alpha * total_shots)), 1)

    accum_energy = 0.0
    accum_shots = 0
    for e, count in records:
        take = min(count, cutoff_shots - accum_shots)
        accum_energy += e * take
        accum_shots += take
        if accum_shots >= cutoff_shots:
            break

    return float(accum_energy / max(accum_shots, 1))
