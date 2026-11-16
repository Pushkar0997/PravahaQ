"""
Egger et al. Warm-Start Quantum State Initialization & Rotated Mixer
====================================================================
Maps continuous relaxation variables c_i* in [0, 1] into regularized initial
single-qubit Ry(theta_i) preparation angles and rotated mixer coefficients:

    theta_i = 2 * arcsin(sqrt(c_i_reg))
    H_M^{(ws)} = sum_i ( -sin(theta_i) X_i - cos(theta_i) Z_i )

Reference:
  Egger, D. J., Marecek, J., & Woerner, S. (2021). "Warm-starting quantum
  optimization." Quantum, 5, 479.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WarmStartSchedule:
    """Regularized warm-start rotation angles and mixer axis coefficients."""

    regularized_c: list[float]
    ry_angles_rad: list[float]
    mixer_x_coeffs: list[float]
    mixer_z_coeffs: list[float]
    epsilon: float


def compute_warmstart_schedule(
    continuous_c: list[float] | np.ndarray,
    epsilon: float = 0.25,
) -> WarmStartSchedule:
    """
    Regularize c_i* in [0, 1] into [epsilon, 1 - epsilon] so no qubit starts in a
    frozen computational basis state (which would trap standard Z-phase gates),
    and compute theta_i = 2 * arcsin(sqrt(c_i_reg)).
    """
    eps = float(np.clip(epsilon, 0.01, 0.49))
    c_reg: list[float] = []
    thetas: list[float] = []
    mx: list[float] = []
    mz: list[float] = []

    for val in continuous_c:
        c = float(np.clip(val, 0.0, 1.0))
        if c <= eps:
            cr = eps
        elif c >= 1.0 - eps:
            cr = 1.0 - eps
        else:
            cr = c

        theta = float(2.0 * np.arcsin(np.sqrt(cr)))
        c_reg.append(round(cr, 6))
        thetas.append(round(theta, 6))
        mx.append(round(float(-np.sin(theta)), 6))
        mz.append(round(float(-np.cos(theta)), 6))

    return WarmStartSchedule(
        regularized_c=c_reg,
        ry_angles_rad=thetas,
        mixer_x_coeffs=mx,
        mixer_z_coeffs=mz,
        epsilon=eps,
    )
