"""
Montanez-Barrera Slack-Free Unbalanced Inequality Penalization
==============================================================
Encodes linear inequality constraints g(x) = C_max - sum_i w_i x_i >= 0 directly into
2-body polynomial coefficients without introducing binary slack ancilla qubits.

Reference:
  Montanez-Barrera, J. A., Willsch, D., & Michielsen, K. (2024).
  "Unbalanced penalization: A new approach to encode inequality constraints of
  combinatorial problems for quantum optimization algorithms."
  Quantum Science and Technology, 9(2), 025022.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InequalityConstraintSpec:
    """
    Represents a bounded resource inequality constraint:
        g(x) = capacity - sum_{i} weights[i] * x_{var_indices[i]} >= 0
    Where weights and capacity are normalized to dimensionless load units.
    """

    name: str
    var_indices: tuple[int, ...]
    weights: tuple[float, ...]
    capacity: float


def compile_unbalanced_inequality_penalty(
    constraint: InequalityConstraintSpec,
    lambda1: float = 3.5,
    lambda2: float = 6.0,
) -> tuple[float, dict[int, float], dict[tuple[int, int], float]]:
    """
    Expand the 2nd-order Taylor unbalanced exponential penalty:
        Phi(x) = -lambda1 * h(x) + lambda2 * h(x)^2
    where h(x) = (capacity - sum_i w_i x_i) / scale is the normalized slack.

    Because x_i in {0, 1} satisfies x_i^2 = x_i, this produces strictly:
      - Constant offset c_0
      - 1-body linear terms L_i * x_i
      - 2-body quadratic terms Q_{ij} * x_i * x_j (for i < j)
    with ZERO slack variables.
    """
    scale = max(abs(constraint.capacity), 1.0)
    c_norm = constraint.capacity / scale
    w_norm = [w / scale for w in constraint.weights]
    idx = constraint.var_indices
    k = len(idx)

    # h(x) = c_norm - sum_a w_a x_a
    # Phi(x) = -lambda1 * c_norm + lambda2 * c_norm^2
    #          + sum_a (lambda1 * w_a - 2 * lambda2 * c_norm * w_a + lambda2 * w_a^2) x_a
    #          + sum_{a < b} (2 * lambda2 * w_a * w_b) x_a x_b
    offset = -lambda1 * c_norm + lambda2 * (c_norm**2)
    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}

    for a in range(k):
        var_a = idx[a]
        wa = w_norm[a]
        coeff_1 = lambda1 * wa - 2.0 * lambda2 * c_norm * wa + lambda2 * (wa**2)
        linear[var_a] = linear.get(var_a, 0.0) + coeff_1

        for b in range(a + 1, k):
            var_b = idx[b]
            wb = w_norm[b]
            pair = (min(var_a, var_b), max(var_a, var_b))
            coeff_2 = 2.0 * lambda2 * wa * wb
            quadratic[pair] = quadratic.get(pair, 0.0) + coeff_2

    return offset, linear, quadratic


def evaluate_unbalanced_penalty_curve(
    slack_values: list[float] | tuple[float, ...],
    lambda1: float = 3.5,
    lambda2: float = 6.0,
) -> list[float]:
    """
    Evaluate Phi(g) = -lambda1 * g + lambda2 * g^2 across a grid of slack values g
    (used for plotting the asymmetric barrier curve in UI and paper figures).
    """
    return [float(-lambda1 * g + lambda2 * (g**2)) for g in slack_values]
