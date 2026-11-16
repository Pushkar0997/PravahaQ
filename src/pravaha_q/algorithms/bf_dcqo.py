"""
Bias-Field Digitized Counterdiabatic Quantum Optimization (BF-DCQO)
===================================================================
Implements the zero-variational-outer-loop counterdiabatic optimization
algorithm (RFC-0002):
  1. Computes analytical Adiabatic Gauge Potential (AGP) impulse angles.
  2. Evolves the quantum state under digitized Cost + Transverse Mixer +
     1-body/2-body/3-body Counterdiabatic Gauge gates.
  3. Measures longitudinal spin magnetizations m_i^{(k)} = <Z_i> and updates
     the longitudinal bias field:
         h_i^{(k+1)} = h_i^{(0)} + eta_k * tanh(gamma_bias * m_i^{(k)})
  4. Converges in 3 to 5 ultra-shallow iterations without classical COBYLA/Adam!

References:
  - Cadavid, A. G., et al. (2025). Phys. Rev. Research, 7(2), L022010.
  - Romero, S. V., et al. (2025). arXiv:2409.04477 (HUBO BF-DCQO).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pravaha_q.algorithms.agp_commutator import (
    CounterdiabaticImpulseSchedule,
    compute_analytical_agp_schedule,
)
from pravaha_q.config import BFDCQOConfig
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass


@dataclass(frozen=True)
class BFDCQOIterationTelemetry:
    """Telemetry recorded at a single bias-field iteration k."""

    iteration: int
    expectation_energy: float
    best_sampled_energy: float
    mean_spin_polarization: float
    magnetizations: list[float]
    alpha1_gauge_coeff: float


@dataclass(frozen=True)
class BFDCQOResult:
    """Complete execution result of the BF-DCQO engine."""

    num_qubits: int
    iterations_executed: int
    converged_early: bool
    best_bitstring: str
    best_energy: float
    final_expectation_energy: float
    final_polarization: float
    bitstring_counts: dict[str, int]
    statevector_probabilities: dict[str, float]
    iteration_history: list[BFDCQOIterationTelemetry] = field(default_factory=list)
    final_agp_schedule: CounterdiabaticImpulseSchedule | None = None


class BFDCQOEngine:
    """Fast vectorized & tensor-axis BF-DCQO engine for 3-local Ising Spin Glasses."""

    def __init__(
        self,
        config: BFDCQOConfig | None = None,
        shots: int = 2048,
        seed: int = 42,
    ) -> None:
        self.config = config or BFDCQOConfig()
        self.shots = shots
        self.seed = seed

    def run(self, ising: IsingSpinGlass) -> BFDCQOResult:
        """Execute the iterative BF-DCQO loop on a 3-local IsingSpinGlass."""
        n = ising.num_qubits
        diag_cost = ising.diagonal_vector()
        scale = max(ising.spectral_scale, 1e-9)
        norm_diag = (diag_cost - ising.constant_offset) / scale

        bias_fields: dict[int, float] = {i: 0.0 for i in range(n)}
        history: list[BFDCQOIterationTelemetry] = []
        best_overall_energy = float("inf")
        best_overall_bitstring = "0" * n
        converged_early = False

        last_probs = np.ones(1 << n, dtype=np.float64) / (1 << n)
        last_schedule: CounterdiabaticImpulseSchedule | None = None

        max_k = max(self.config.max_iterations, 1)
        for k in range(max_k):
            lam_k = min(0.45 + 0.12 * k, 0.85)
            dt_k = self.config.impulse_dt * (1.0 + 0.15 * k)

            agp_sched = compute_analytical_agp_schedule(
                ising=ising,
                bias_fields=bias_fields,
                lambda_point=lam_k,
                impulse_dt=dt_k,
            )
            last_schedule = agp_sched

            # Simulate single-layer counterdiabatic + bias-field impulse circuit
            state = self._evolve_bf_dcqo_step(
                num_qubits=n,
                norm_diag=norm_diag,
                bias_fields=bias_fields,
                scale=scale,
                agp_schedule=agp_sched,
                cost_gamma=0.55 * dt_k,
                mixer_beta=0.35 * (1.0 - lam_k),
            )

            probs = np.abs(state) ** 2
            probs = probs / np.sum(probs)
            last_probs = probs

            exp_energy = float(np.sum(probs * diag_cost))
            mags = compute_z_magnetizations(probs, n)
            mean_pol = float(np.mean(np.abs(mags)))

            # Identify lowest energy among high-probability states & greedy magnetizations
            mag_bits = "".join("0" if m >= 0.0 else "1" for m in mags)
            mag_e = ising.evaluate_bitstring(mag_bits)

            top_indices = np.argsort(probs)[-min(16, 1 << n) :]
            iter_best_e = mag_e
            iter_best_bits = mag_bits
            for idx in top_indices:
                e_idx = float(diag_cost[int(idx)])
                if e_idx < iter_best_e:
                    iter_best_e = e_idx
                    iter_best_bits = format(int(idx), f"0{n}b")

            if iter_best_e < best_overall_energy:
                best_overall_energy = iter_best_e
                best_overall_bitstring = iter_best_bits

            history.append(
                BFDCQOIterationTelemetry(
                    iteration=k + 1,
                    expectation_energy=round(exp_energy, 6),
                    best_sampled_energy=round(iter_best_e, 6),
                    mean_spin_polarization=round(mean_pol, 5),
                    magnetizations=[round(float(m), 5) for m in mags],
                    alpha1_gauge_coeff=agp_sched.alpha1_coeff,
                )
            )

            if mean_pol >= self.config.polarization_stop_threshold and k >= 1:
                converged_early = True
                break

            # RFC-0002 Sigmoidal Bias-Field Ratchet Update
            eta_k = (
                self.config.initial_bias_gain
                * scale
                * (1.0 + (k + 1) / max_k)
            )
            gamma_b = self.config.bias_sharpness_gamma
            for i in range(n):
                bias_fields[i] = float(eta_k * np.tanh(gamma_b * mags[i]))

        # Sample shot histogram from final quantum state distribution
        counts = sample_bitstring_counts(
            last_probs, num_qubits=n, shots=self.shots, seed=self.seed
        )
        for bitstr in counts:
            e_b = ising.evaluate_bitstring(bitstr)
            if e_b < best_overall_energy:
                best_overall_energy = e_b
                best_overall_bitstring = bitstr

        top_prob_indices = np.argsort(last_probs)[-min(12, 1 << n) :][::-1]
        prob_dict = {
            format(int(idx), f"0{n}b"): round(float(last_probs[int(idx)]), 5)
            for idx in top_prob_indices
            if last_probs[int(idx)] > 1e-4
        }

        return BFDCQOResult(
            num_qubits=n,
            iterations_executed=len(history),
            converged_early=converged_early,
            best_bitstring=best_overall_bitstring,
            best_energy=round(best_overall_energy, 6),
            final_expectation_energy=history[-1].expectation_energy,
            final_polarization=history[-1].mean_spin_polarization,
            bitstring_counts=counts,
            statevector_probabilities=prob_dict,
            iteration_history=history,
            final_agp_schedule=last_schedule,
        )

    @staticmethod
    def _evolve_bf_dcqo_step(
        num_qubits: int,
        norm_diag: np.ndarray,
        bias_fields: dict[int, float],
        scale: float,
        agp_schedule: CounterdiabaticImpulseSchedule,
        cost_gamma: float,
        mixer_beta: float,
    ) -> np.ndarray:
        """
        Evolve |+>^{otimes n} (tilted by longitudinal bias fields) through
        Cost Phase + Counterdiabatic Y/YZ/YZZ Gauge Impulse + Transverse X Mixer.
        """
        n = num_qubits
        dim = 1 << n
        # 1. Start in uniform superposition |+>^{otimes n}
        state = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
        psi = state.reshape((2,) * n)

        # 2. Apply initial longitudinal bias-field tilt Ry(theta_bias_i)
        for i, b_val in bias_fields.items():
            if abs(b_val) > 1e-12:
                # Positive bias field (+Z) favors |0>, negative (-Z) favors |1>
                tilt = float(-0.65 * np.arctan(b_val / max(scale, 1e-9)))
                psi = apply_single_qubit_ry(psi, i, tilt)

        # 3. Apply 1-body Counterdiabatic Y-impulse exp(-i theta_i Y_i) == Ry(2 * theta_i)
        for i, y_angle in agp_schedule.y_1body_angles.items():
            psi = apply_single_qubit_ry(psi, i, 2.0 * y_angle)

        # 4. Apply 2-body Counterdiabatic (Y_i Z_j + Z_i Y_j) impulse via basis rotation
        for (i, j), yz_angle in agp_schedule.yz_2body_angles.items():
            psi = apply_two_qubit_yz_zy(psi, i, j, yz_angle, n)

        # 5. Apply 3-body Counterdiabatic (Y_i Z_j Z_k + sym) impulse
        for (i, j, k), yzz_angle in agp_schedule.yzz_3body_angles.items():
            psi = apply_three_qubit_yzz(psi, i, j, k, yzz_angle, n)

        # 6. Apply exact diagonal Cost + Longitudinal Bias Phase exp(-i gamma H_C)
        flat_psi = psi.reshape(-1)
        flat_psi = flat_psi * np.exp(-1.0j * cost_gamma * norm_diag)
        psi = flat_psi.reshape((2,) * n)

        # 7. Apply Transverse X-Mixer exp(i beta X_i) == Rx(-2 * beta)
        for i in range(n):
            psi = apply_single_qubit_rx(psi, i, -2.0 * mixer_beta)

        return psi.reshape(-1)


# --- High-Speed Tensor-Axis Gate Primitives ---


def apply_single_qubit_ry(psi: np.ndarray, qubit: int, angle: float) -> np.ndarray:
    """Apply Ry(angle) = exp(-i angle/2 Y) to axis `qubit` of an n-qubit tensor."""
    c = np.cos(angle / 2.0)
    s = np.sin(angle / 2.0)
    mat = np.array([[c, -s], [s, c]], dtype=np.complex128)
    out = np.tensordot(mat, psi, axes=([1], [qubit]))
    return np.moveaxis(out, 0, qubit)


def apply_single_qubit_rx(psi: np.ndarray, qubit: int, angle: float) -> np.ndarray:
    """Apply Rx(angle) = exp(-i angle/2 X) to axis `qubit` of an n-qubit tensor."""
    c = np.cos(angle / 2.0)
    s = -1.0j * np.sin(angle / 2.0)
    mat = np.array([[c, s], [s, c]], dtype=np.complex128)
    out = np.tensordot(mat, psi, axes=([1], [qubit]))
    return np.moveaxis(out, 0, qubit)


def apply_two_qubit_yz_zy(
    psi: np.ndarray, q1: int, q2: int, theta: float, num_qubits: int
) -> np.ndarray:
    """Apply exp(-i theta (Y_{q1} Z_{q2} + Z_{q1} Y_{q2})) via Rx(pi/2) conjugation."""
    # Y_{q1} Z_{q2} = Rx_{q1}(pi/2) Z_{q1} Z_{q2} Rx_{q1}(-pi/2)
    z_pair = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    phase = np.exp(-1.0j * theta * z_pair)
    shape = [1] * num_qubits
    shape[q1] = 2
    shape[q2] = 2
    phase_tensor = phase.reshape(shape)

    psi = apply_single_qubit_rx(psi, q1, np.pi / 2.0)
    psi = psi * phase_tensor
    psi = apply_single_qubit_rx(psi, q1, -np.pi / 2.0)

    psi = apply_single_qubit_rx(psi, q2, np.pi / 2.0)
    psi = psi * phase_tensor
    psi = apply_single_qubit_rx(psi, q2, -np.pi / 2.0)
    return psi


def apply_three_qubit_yzz(
    psi: np.ndarray, q1: int, q2: int, q3: int, theta: float, num_qubits: int
) -> np.ndarray:
    """Apply exp(-i theta Y_{q1} Z_{q2} Z_{q3}) via Rx(pi/2) basis rotation."""
    z1 = np.array([1.0, -1.0], dtype=np.float64)
    z_triad = np.einsum("i,j,k->ijk", z1, z1, z1)
    phase = np.exp(-1.0j * theta * z_triad)
    shape = [1] * num_qubits
    shape[q1] = 2
    shape[q2] = 2
    shape[q3] = 2
    phase_tensor = phase.reshape(shape)

    psi = apply_single_qubit_rx(psi, q1, np.pi / 2.0)
    psi = psi * phase_tensor
    psi = apply_single_qubit_rx(psi, q1, -np.pi / 2.0)
    return psi


def compute_z_magnetizations(probs: np.ndarray, num_qubits: int) -> np.ndarray:
    """Compute exact single-qubit expectations m_i = <Z_i> from probability vector."""
    dim = 1 << num_qubits
    indices = np.arange(dim, dtype=np.uint32)
    mags = np.empty(num_qubits, dtype=np.float64)
    for q in range(num_qubits):
        shift = num_qubits - 1 - q
        bit = (indices >> shift) & 1
        spins = 1.0 - 2.0 * bit
        mags[q] = float(np.sum(probs * spins))
    return mags


def sample_bitstring_counts(
    probs: np.ndarray, num_qubits: int, shots: int = 2048, seed: int = 42
) -> dict[str, int]:
    """Sample finite-shot measurement histogram from a state probability distribution."""
    rng = np.random.default_rng(seed)
    dim = probs.shape[0]
    p_clean = np.clip(probs, 0.0, 1.0)
    p_clean = p_clean / np.sum(p_clean)
    samples = rng.choice(dim, size=shots, p=p_clean)
    unique, counts = np.unique(samples, return_counts=True)
    order = np.argsort(counts)[::-1]
    return {
        format(int(unique[i]), f"0{num_qubits}b"): int(counts[i]) for i in order
    }
