"""
Zero-Noise Extrapolation (ZNE) via Unitary Folding & Richardson Extrapolation
================================================================================
Scales circuit gate depth by scale factors c in {1, 3, 5} via Richardson unitary
folding (U -> U U^dagger U) and extrapolates back to the zero-noise limit c -> 0.
"""

from __future__ import annotations

import numpy as np


def richardson_zero_noise_extrapolation(
    scale_factors: tuple[float, ...] = (1.0, 3.0, 5.0),
    measured_expectations: tuple[float, ...] = (10.0, 12.5, 15.2),
) -> float:
    """
    Compute Richardson polynomial extrapolation estimate at scale factor c = 0.
    """
    c = np.array(scale_factors, dtype=np.float64)
    y = np.array(measured_expectations, dtype=np.float64)
    m = len(c)

    # Lagrange basis polynomial evaluated at c = 0
    weights = np.empty(m, dtype=np.float64)
    for i in range(m):
        prod = 1.0
        for j in range(m):
            if i != j:
                prod *= c[j] / (c[j] - c[i])
        weights[i] = prod

    extrapolated_val = float(np.sum(weights * y))
    return extrapolated_val
