"""
Continuous Box-Constrained Quadratic/Cubic Relaxation Solver
============================================================
Relaxes binary decision variables x in {0, 1}^N to the continuous hypercube
c in [0, 1]^N and minimizes the smooth HUBO energy functional using L-BFGS-B
with exact analytical gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from pravaha_q.math_engine.hubo_builder import HUBOPolynomial


@dataclass(frozen=True)
class QPRelaxationResult:
    """Continuous relaxation vector c* in [0, 1]^N and rounded binary projection."""

    continuous_solution: list[float]
    rounded_bitstring: str
    relaxed_energy: float
    rounded_energy: float
    iterations: int


def solve_box_relaxed_hubo(
    hubo: HUBOPolynomial,
    num_restarts: int = 3,
    seed: int = 42,
) -> QPRelaxationResult:
    """
    Minimize E_HUBO(c) subject to c_i in [0, 1] using multi-start L-BFGS-B
    with exact analytical partial derivatives dE / dc_i.
    """
    n = hubo.num_variables
    rng = np.random.default_rng(seed)
    bounds = [(0.0, 1.0)] * n

    def energy_and_grad(c: np.ndarray) -> tuple[float, np.ndarray]:
        val = float(hubo.constant_offset)
        grad = np.zeros(n, dtype=np.float64)

        for i, l_coeff in hubo.linear.items():
            val += l_coeff * c[i]
            grad[i] += l_coeff

        for (i, j), q_coeff in hubo.quadratic.items():
            val += q_coeff * c[i] * c[j]
            grad[i] += q_coeff * c[j]
            grad[j] += q_coeff * c[i]

        for (i, j, k), t_coeff in hubo.cubic.items():
            val += t_coeff * c[i] * c[j] * c[k]
            grad[i] += t_coeff * c[j] * c[k]
            grad[j] += t_coeff * c[i] * c[k]
            grad[k] += t_coeff * c[i] * c[j]

        return val, grad

    best_c = np.full(n, 0.5, dtype=np.float64)
    best_val = float("inf")
    total_nit = 0

    for r in range(num_restarts):
        x0 = (
            np.full(n, 0.5, dtype=np.float64)
            if r == 0
            else rng.uniform(0.15, 0.85, size=n)
        )
        res = minimize(
            energy_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 75, "ftol": 1e-8},
        )
        total_nit += int(getattr(res, "nit", 1))
        if float(res.fun) < best_val:
            best_val = float(res.fun)
            best_c = np.clip(res.x, 0.0, 1.0)

    rounded_bits = [1 if val >= 0.5 else 0 for val in best_c]
    rounded_str = "".join(str(b) for b in rounded_bits)
    rounded_e = hubo.evaluate(rounded_bits)

    return QPRelaxationResult(
        continuous_solution=[round(float(v), 6) for v in best_c],
        rounded_bitstring=rounded_str,
        relaxed_energy=round(best_val, 6),
        rounded_energy=round(rounded_e, 6),
        iterations=total_nit,
    )
