"""Scientific benchmarking metrics (Approximation Ratio, TTS_99, CNOT Count)."""

from __future__ import annotations

import numpy as np


def compute_approximation_ratio(
    best_energy: float, exact_ground_energy: float, max_energy: float
) -> float:
    """Compute standard Approximation Ratio AR = (E_max - E_found) / (E_max - E_0)."""
    if abs(max_energy - exact_ground_energy) < 1e-9:
        return 1.0
    ar = (max_energy - best_energy) / max(max_energy - exact_ground_energy, 1e-9)
    return float(np.clip(ar, 0.0, 1.0))
