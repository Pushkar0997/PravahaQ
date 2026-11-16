"""
Multistage Pauli Correlation Encoding (PCE) for Sub-Linear Qubit Compression
============================================================================
Encodes N binary decision variables onto n << N physical qubits by mapping each
variable m in {0, ..., N-1} to the expectation value of a distinct k-body Pauli
correlator Pi_m in {I, X, Y, Z}^{otimes n}:

    z_tilde_m(theta) = tanh(alpha_s * <psi(theta)| Pi_m |psi(theta)>)

Optimizes a hardware-efficient brickwall variational circuit across a multistage
(alpha_s, beta_s) annealing curriculum to prevent premature vanishing gradients.

Reference:
  Usuki, S., et al. (2026). "Multistage Pauli Correlation Encoding for Scalable
  Quantum Optimization." arXiv:2609.10665.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from pravaha_q.domain.exceptions import QubitCompressionCapacityError
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass


@dataclass(frozen=True)
class PCECurriculumStage:
    """Hyperparameters for a single curriculum annealing stage in Multistage PCE."""

    stage_id: int
    alpha_slope: float
    beta_reg: float
    max_steps: int = 25


@dataclass(frozen=True)
class PCEResult:
    """Telemetry and decoded solution returned by Multistage PCE."""

    num_variables: int
    num_qubits_used: int
    compression_ratio: float
    best_bitstring: str
    best_energy: float
    correlator_strings: list[str]
    final_correlator_expectations: list[float]
    loss_history: list[float] = field(default_factory=list)
    optimal_angles: list[float] = field(default_factory=list)


def generate_pauli_correlator_basis(
    num_qubits: int,
    num_variables: int,
    max_body_k: int = 3,
) -> list[tuple[tuple[int, str], ...]]:
    """
    Generate `num_variables` distinct k-body Pauli observables (1 <= k <= max_body_k)
    over `num_qubits` physical qubits.

    Capacity for n qubits:
      - k=1: 3 * n
      - k=2: 3 * C(n, 2) homogeneous (ZZ, XX, YY) + mixed
      - k=3: 3 * C(n, 3) homogeneous (ZZZ, XXX, YYY)
    For n=7 qubits, homogeneous k in {1,2,3} alone provides 21 + 63 + 105 = 189 variables.
    """
    basis: list[tuple[tuple[int, str], ...]] = []

    # Prioritize k=2 and k=3 homogeneous correlators (ZZ, XX, YY, ZZZ, XXX, YYY)
    for k in range(2, max_body_k + 1):
        for pauli_char in ("Z", "X", "Y"):
            for sites in combinations(range(num_qubits), k):
                basis.append(tuple((q, pauli_char) for q in sites))
                if len(basis) >= num_variables:
                    return basis

    # Add k=1 single-qubit observables if needed
    for pauli_char in ("Z", "X", "Y"):
        for q in range(num_qubits):
            basis.append(((q, pauli_char),))
            if len(basis) >= num_variables:
                return basis

    # Add mixed ZX / ZY 2-body correlators if still needed
    for q1, q2 in combinations(range(num_qubits), 2):
        for p1, p2 in (("Z", "X"), ("X", "Z"), ("Z", "Y"), ("Y", "Z")):
            basis.append(((q1, p1), (q2, p2)))
            if len(basis) >= num_variables:
                return basis

    raise QubitCompressionCapacityError(
        f"Cannot encode N={num_variables} variables onto n={num_qubits} qubits "
        f"with max_body_k={max_body_k} (max capacity generated={len(basis)})."
    )


def format_correlator_string(
    correlator: tuple[tuple[int, str], ...], num_qubits: int
) -> str:
    """Format sparse correlator tuple into a dense Pauli string like 'IZZIIZ'."""
    chars = ["I"] * num_qubits
    for q, p in correlator:
        chars[q] = p
    return "".join(chars)


class MultistagePCEEncoder:
    """
    Executes Multistage Pauli Correlation Encoding on an IsingSpinGlass of N variables
    compressed onto n << N qubits.
    """

    def __init__(
        self,
        num_qubits: int = 6,
        ansatz_layers: int = 3,
        max_body_k: int = 3,
        stages: tuple[PCECurriculumStage, ...] | None = None,
        seed: int = 42,
    ) -> None:
        self.num_qubits = num_qubits
        self.ansatz_layers = ansatz_layers
        self.max_body_k = max_body_k
        self.stages = stages or (
            PCECurriculumStage(stage_id=1, alpha_slope=1.5, beta_reg=0.08, max_steps=25),
            PCECurriculumStage(stage_id=2, alpha_slope=3.5, beta_reg=0.20, max_steps=25),
            PCECurriculumStage(stage_id=3, alpha_slope=7.5, beta_reg=0.45, max_steps=20),
        )
        self.seed = seed

    def solve(self, ising: IsingSpinGlass) -> PCEResult:
        """Compress and solve an N-variable IsingSpinGlass on self.num_qubits qubits."""
        N = ising.num_qubits
        n = self.num_qubits
        correlators = generate_pauli_correlator_basis(
            num_qubits=n, num_variables=N, max_body_k=self.max_body_k
        )

        # Precompute bit-flip and phase masks for fast expectation <psi| Pi_m |psi>
        masks = [self._compile_pauli_mask(corr, n) for corr in correlators]

        num_params = 2 * n * self.ansatz_layers
        rng = np.random.default_rng(self.seed)
        theta = rng.uniform(-0.4, 0.4, size=num_params)

        loss_history: list[float] = []
        scale = max(ising.spectral_scale, 1.0)

        for stage in self.stages:
            alpha = stage.alpha_slope
            beta = stage.beta_reg

            def objective(params: np.ndarray) -> float:
                state = self._simulate_brickwall_ansatz(params, n, self.ansatz_layers)
                exp_vals = self._evaluate_all_correlators(state, masks)
                z_tilde = np.tanh(alpha * exp_vals)
                e_relaxed = self._evaluate_relaxed_ising(ising, z_tilde) / scale
                sign_reg = beta * float(np.mean(1.0 - z_tilde**2))
                val = e_relaxed + sign_reg
                loss_history.append(float(val * scale))
                return val

            res = minimize(
                objective,
                theta,
                method="COBYLA",
                options={"maxiter": stage.max_steps, "rhobeg": 0.35},
            )
            theta = res.x

        # Final readout and sign projection z_m* = sgn(<Pi_m>)
        final_state = self._simulate_brickwall_ansatz(theta, n, self.ansatz_layers)
        final_exps = self._evaluate_all_correlators(final_state, masks)

        # Evaluate primary sign projection + top single-variable local flips
        spins_hard = [1 if val >= 0.0 else -1 for val in final_exps]
        bits_hard = [(1 - s) // 2 for s in spins_hard]
        best_bits = list(bits_hard)
        best_energy = ising.evaluate_bitstring(best_bits)

        # Polish uncertain correlators (smallest |<Pi_m>|)
        uncertainty_order = np.argsort(np.abs(final_exps))[: min(N, 12)]
        for idx in uncertainty_order:
            candidate = list(best_bits)
            candidate[int(idx)] ^= 1
            cand_e = ising.evaluate_bitstring(candidate)
            if cand_e < best_energy:
                best_energy = cand_e
                best_bits = candidate

        best_bitstring = "".join(str(b) for b in best_bits)
        corr_strs = [format_correlator_string(c, n) for c in correlators]

        return PCEResult(
            num_variables=N,
            num_qubits_used=n,
            compression_ratio=round(N / max(n, 1), 3),
            best_bitstring=best_bitstring,
            best_energy=float(best_energy),
            correlator_strings=corr_strs,
            final_correlator_expectations=[round(float(x), 5) for x in final_exps],
            loss_history=loss_history,
            optimal_angles=[round(float(a), 5) for a in theta],
        )

    @staticmethod
    def _simulate_brickwall_ansatz(
        params: np.ndarray, num_qubits: int, layers: int
    ) -> np.ndarray:
        """
        Simulate a hardware-efficient Ry-Rz + CNOT brickwall ansatz on n qubits
        using tensor-axis contractions in O(layers * n * 2^n) time.
        """
        state = np.zeros(1 << num_qubits, dtype=np.complex128)
        state[0] = 1.0 + 0.0j
        psi = state.reshape((2,) * num_qubits)

        p_idx = 0
        for layer in range(layers):
            # Single-qubit Ry and Rz rotations
            for q in range(num_qubits):
                ry_angle = float(params[p_idx])
                rz_angle = float(params[p_idx + 1])
                p_idx += 2

                c_y = np.cos(ry_angle / 2.0)
                s_y = np.sin(ry_angle / 2.0)
                ry_mat = np.array([[c_y, -s_y], [s_y, c_y]], dtype=np.complex128)
                psi = np.tensordot(ry_mat, psi, axes=([1], [q]))
                psi = np.moveaxis(psi, 0, q)

                e_m = np.exp(-0.5j * rz_angle)
                e_p = np.exp(0.5j * rz_angle)
                rz_vec = np.array([e_m, e_p], dtype=np.complex128)
                shape = [1] * num_qubits
                shape[q] = 2
                psi = psi * rz_vec.reshape(shape)

            # Entangling CNOT ring
            offset = layer % 2
            for q in range(offset, num_qubits - 1, 2):
                psi = _apply_cnot_tensor(psi, q, q + 1, num_qubits)
            if num_qubits > 2:
                psi = _apply_cnot_tensor(psi, num_qubits - 1, 0, num_qubits)

        return psi.reshape(-1)

    @staticmethod
    def _compile_pauli_mask(
        correlator: tuple[tuple[int, str], ...], num_qubits: int
    ) -> tuple[int, np.ndarray]:
        """Precompute bit-flip mask and phase vector for fast <psi| Pi_m |psi>."""
        dim = 1 << num_qubits
        indices = np.arange(dim, dtype=np.uint32)
        flip_mask = 0
        phase = np.ones(dim, dtype=np.complex128)

        for q, p in correlator:
            bit_pos = num_qubits - 1 - q
            bit_val = (indices >> bit_pos) & 1
            if p == "Z":
                phase *= np.where(bit_val == 0, 1.0, -1.0)
            elif p == "X":
                flip_mask |= 1 << bit_pos
            elif p == "Y":
                flip_mask |= 1 << bit_pos
                phase *= np.where(bit_val == 0, 1.0j, -1.0j)

        return flip_mask, phase

    @staticmethod
    def _evaluate_all_correlators(
        state: np.ndarray, masks: list[tuple[int, np.ndarray]]
    ) -> np.ndarray:
        """Compute <psi| Pi_m |psi> for all m in 1..N."""
        dim = state.shape[0]
        indices = np.arange(dim, dtype=np.uint32)
        conj_state = np.conj(state)
        out = np.empty(len(masks), dtype=np.float64)
        for m_idx, (flip_mask, phase) in enumerate(masks):
            if flip_mask == 0:
                out[m_idx] = float(np.real(np.sum(conj_state * phase * state)))
            else:
                permuted = state[indices ^ flip_mask]
                out[m_idx] = float(np.real(np.sum(conj_state * phase * permuted)))
        return out

    @staticmethod
    def _evaluate_relaxed_ising(ising: IsingSpinGlass, z_tilde: np.ndarray) -> float:
        """Evaluate relaxed continuous Ising energy E(z_tilde)."""
        e = float(ising.constant_offset)
        for i, h_val in ising.h_linear.items():
            e += h_val * float(z_tilde[i])
        for (i, j), j_val in ising.j_quadratic.items():
            e += j_val * float(z_tilde[i] * z_tilde[j])
        for (i, j, k), k_val in ising.k_cubic.items():
            e += k_val * float(z_tilde[i] * z_tilde[j] * z_tilde[k])
        return e


def _apply_cnot_tensor(
    psi: np.ndarray, control: int, target: int, num_qubits: int
) -> np.ndarray:
    """Apply CNOT gate between control and target axes of an n-qubit state tensor."""
    out = psi.copy()
    idx_ctrl_1 = [slice(None)] * num_qubits
    idx_ctrl_1[control] = 1

    idx_flipped = list(idx_ctrl_1)
    idx_flipped[target] = slice(None, None, -1)

    out[tuple(idx_ctrl_1)] = psi[tuple(idx_flipped)]
    return out
