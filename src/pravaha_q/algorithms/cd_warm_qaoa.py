"""
Counterdiabatic Warm-Started QAOA (CD-WarmQAOA)
===============================================
Combines continuous QP/SDP warm-start initialization |psi_0> = bigotimes_i Ry(theta_i)|0>,
Egger et al. rotated warm-start mixer, and 1-local/2-local counterdiabatic
driving terms optimized via COBYLA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from pravaha_q.algorithms.bf_dcqo import (
    apply_single_qubit_rx,
    apply_single_qubit_ry,
    apply_two_qubit_yz_zy,
    sample_bitstring_counts,
)
from pravaha_q.math_engine.hubo_builder import HUBOPolynomial
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass
from pravaha_q.warmstart.qp_box_relaxation import solve_box_relaxed_hubo
from pravaha_q.warmstart.rotation_preconditioner import (
    WarmStartSchedule,
    compute_warmstart_schedule,
)


@dataclass(frozen=True)
class CDWarmQAOAResult:
    """Result and convergence trace for CD-WarmQAOA."""

    num_qubits: int
    reps_p: int
    best_bitstring: str
    best_energy: float
    final_expectation_energy: float
    warmstart_schedule: WarmStartSchedule
    optimal_parameters: list[float]
    energy_trace: list[float] = field(default_factory=list)
    bitstring_counts: dict[str, int] = field(default_factory=dict)


class CDWarmQAOASolver:
    """Executes Counterdiabatic Warm-Started QAOA at depth p."""

    def __init__(
        self,
        reps_p: int = 2,
        warmstart_epsilon: float = 0.25,
        maxiter: int = 45,
        shots: int = 2048,
        seed: int = 42,
    ) -> None:
        self.reps_p = reps_p
        self.warmstart_epsilon = warmstart_epsilon
        self.maxiter = maxiter
        self.shots = shots
        self.seed = seed

    def solve(
        self, ising: IsingSpinGlass, hubo: HUBOPolynomial | None = None
    ) -> CDWarmQAOAResult:
        """Warm-start from continuous relaxation and optimize (gamma, beta, alpha_cd)."""
        n = ising.num_qubits
        if hubo is not None:
            qp_res = solve_box_relaxed_hubo(hubo, seed=self.seed)
            c_init = qp_res.continuous_solution
        else:
            # Fallback heuristic from linear fields h_i
            scale_h = max(
                (abs(v) for v in ising.h_linear.values()), default=1.0
            )
            c_init = [
                float(0.5 + 0.35 * np.tanh(ising.h_linear.get(i, 0.0) / scale_h))
                for i in range(n)
            ]

        ws_sched = compute_warmstart_schedule(
            c_init, epsilon=self.warmstart_epsilon
        )
        diag_cost = ising.diagonal_vector()
        scale = max(ising.spectral_scale, 1e-9)
        norm_diag = (diag_cost - ising.constant_offset) / scale
        h_norm, j_norm, _ = ising.normalized_coefficients()

        # 3 parameters per layer p: (gamma_l, beta_l, alpha_cd_l)
        init_params: list[float] = []
        for layer in range(self.reps_p):
            frac = (layer + 1) / (self.reps_p + 1)
            init_params.extend([0.45 * frac, 0.35 * (1.0 - frac), 0.25])

        energy_trace: list[float] = []

        def simulate_state(params: np.ndarray) -> np.ndarray:
            state = np.zeros(1 << n, dtype=np.complex128)
            state[0] = 1.0 + 0.0j
            psi = state.reshape((2,) * n)

            # Warm-start state preparation: bigotimes_i Ry(theta_i) |0>
            for i, theta_i in enumerate(ws_sched.ry_angles_rad):
                psi = apply_single_qubit_ry(psi, i, theta_i)

            for layer in range(self.reps_p):
                gamma_l = float(params[3 * layer])
                beta_l = float(params[3 * layer + 1])
                alpha_l = float(params[3 * layer + 2])

                # 1. Cost unitary exp(-i gamma_l H_C)
                flat = psi.reshape(-1) * np.exp(-1.0j * gamma_l * norm_diag)
                psi = flat.reshape((2,) * n)

                # 2. Counterdiabatic 1-body & 2-body gauge layer
                for i, h_val in h_norm.items():
                    psi = apply_single_qubit_ry(psi, i, 2.0 * alpha_l * h_val)
                for (i, j), j_val in j_norm.items():
                    if abs(j_val) > 0.15:
                        psi = apply_two_qubit_yz_zy(
                            psi, i, j, 0.5 * alpha_l * j_val, n
                        )

                # 3. Rotated Warm-Start Mixer: Ry(theta_i) Rz(-2 beta_l) Ry(-theta_i)
                for i, theta_i in enumerate(ws_sched.ry_angles_rad):
                    psi = apply_single_qubit_ry(psi, i, -theta_i)
                    psi = apply_single_qubit_rx(psi, i, -2.0 * beta_l)
                    psi = apply_single_qubit_ry(psi, i, theta_i)

            return psi.reshape(-1)

        def cost_fn(params: np.ndarray) -> float:
            st = simulate_state(params)
            probs = np.abs(st) ** 2
            val = float(np.sum(probs * diag_cost))
            energy_trace.append(round(val, 6))
            return val

        res = minimize(
            cost_fn,
            np.array(init_params, dtype=np.float64),
            method="COBYLA",
            options={"maxiter": self.maxiter, "rhobeg": 0.25},
        )

        final_state = simulate_state(res.x)
        final_probs = np.abs(final_state) ** 2
        final_probs = final_probs / np.sum(final_probs)
        counts = sample_bitstring_counts(
            final_probs, num_qubits=n, shots=self.shots, seed=self.seed
        )

        best_bits = "0" * n
        best_e = float("inf")
        for bstr in counts:
            eb = ising.evaluate_bitstring(bstr)
            if eb < best_e:
                best_e = eb
                best_bits = bstr

        return CDWarmQAOAResult(
            num_qubits=n,
            reps_p=self.reps_p,
            best_bitstring=best_bits,
            best_energy=round(best_e, 6),
            final_expectation_energy=round(float(res.fun), 6),
            warmstart_schedule=ws_sched,
            optimal_parameters=[round(float(v), 5) for v in res.x],
            energy_trace=energy_trace,
            bitstring_counts=counts,
        )
