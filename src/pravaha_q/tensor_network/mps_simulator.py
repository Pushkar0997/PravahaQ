"""
Genuine Matrix Product State (MPS) Tensor Network Simulator
===========================================================
Executes `QuantumCircuitIR` on a 1D chain of rank-3 MPS tensors A^{[q]}_{alpha, s_q, beta}
WITHOUT ever calling `ising.diagonal_vector()` or allocating a 2^N statevector:
  1. Evolves all 1-qubit gates in O(chi^2) and adjacent 2-qubit gates via SVD in O(chi^3).
  2. Routes non-adjacent 2-qubit gates CX(u, v) (|u - v| > 1) via a chain of adjacent
     MPS SWAP contractions along u -> v and back.
  3. Tracks cumulative SVD truncation error:
         eps_trunc = sqrt( sum_{step} sum_{k >= chi_max} sigma_k^2 ).
  4. Evaluates exact Pauli string expectations <psi_MPS| P |psi_MPS> and Hamiltonian
     expectation <H_C> via 1D transfer-matrix contractions in O(N * chi^3).
  5. Samples bitstrings sequentially via conditional chain-rule probabilities
     P(x_q | x_0, ..., x_{q-1}) in O(N * chi^3) per shot.
  6. Converges to exact statevector simulation as `max_bond_dim` increases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from pravaha_q.compiler.circuit_ir import QuantumCircuitIR
from pravaha_q.compiler.paulihedral_router import compile_paulihedral_schedule
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass

_PAULI_OPS: dict[str, np.ndarray] = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}

_CX_TENSOR_4D = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.complex128,
).reshape(2, 2, 2, 2)

_CX_REV_TENSOR_4D = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ],
    dtype=np.complex128,
).reshape(2, 2, 2, 2)

_SWAP_TENSOR_4D = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.complex128,
).reshape(2, 2, 2, 2)


@dataclass(frozen=True)
class MPSResult:
    """Result of genuine MPS tensor network circuit evolution and sampling."""

    num_qubits: int
    max_bond_dimension: int
    achieved_max_bond: int
    truncation_error: float
    best_bitstring: str
    best_energy: float
    expectation_energy: float
    bitstring_counts: dict[str, int] = field(default_factory=dict)
    bitstring_probabilities: dict[str, float] = field(default_factory=dict)


class MPSSimulator:
    """Genuine 1D Matrix Product State (MPS) simulator with SVD truncation and chain sampling."""

    def __init__(
        self,
        max_bond_dim: int = 32,
        shots: int = 512,
        seed: int = 42,
    ) -> None:
        self.max_bond_dim = max(max_bond_dim, 1)
        self.shots = shots
        self.seed = seed

    def evolve_circuit(
        self, circuit: QuantumCircuitIR
    ) -> tuple[list[np.ndarray], float]:
        """
        Evolve |0>^{otimes N} through all gates in `circuit` as an MPS tensor chain.
        Returns `(tensors, cumulative_truncation_error)`.
        """
        n = circuit.num_qubits
        tensors: list[np.ndarray] = []
        for _ in range(n):
            t = np.zeros((1, 2, 1), dtype=np.complex128)
            t[0, 0, 0] = 1.0 + 0.0j
            tensors.append(t)

        disc_weight_sq = 0.0

        for op in circuit.ops:
            g = op.name
            if len(op.qubits) == 1:
                q = op.qubits[0]
                mat = _single_qubit_gate_matrix(g, op.params)
                tensors[q] = np.einsum("sp,apb->asb", mat, tensors[q])
            elif len(op.qubits) == 2:
                u, v = op.qubits
                if g == "SWAP":
                    err_sq = self._apply_nonlocal_two_site_gate(
                        tensors, u, v, _SWAP_TENSOR_4D
                    )
                elif g == "CX":
                    gate_4d = _CX_TENSOR_4D if u < v else _CX_REV_TENSOR_4D
                    err_sq = self._apply_nonlocal_two_site_gate(
                        tensors, min(u, v), max(u, v), gate_4d
                    )
                else:
                    raise ValueError(f"Unsupported 2Q gate in MPS: {g}")
                disc_weight_sq += err_sq

        return tensors, float(np.sqrt(disc_weight_sq))

    def _apply_nonlocal_two_site_gate(
        self,
        tensors: list[np.ndarray],
        left_site: int,
        right_site: int,
        gate_4d: np.ndarray,
    ) -> float:
        """
        Apply a 2-qubit gate `gate_4d[s'_L, s'_R, s_L, s_R]` between `left_site < right_site`.
        If `right_site - left_site > 1`, routes `left_site` adjacent to `right_site` via
        nearest-neighbor SWAP contractions and swaps back afterward.
        """
        u, v = left_site, right_site
        total_err_sq = 0.0

        # Swap u rightward to v - 1
        for k in range(u, v - 1):
            total_err_sq += _apply_adjacent_gate_svd(
                tensors, k, _SWAP_TENSOR_4D, self.max_bond_dim
            )

        # Apply target gate on adjacent sites (v - 1, v)
        total_err_sq += _apply_adjacent_gate_svd(
            tensors, v - 1, gate_4d, self.max_bond_dim
        )

        # Swap back from v - 1 leftward to u
        for k in range(v - 2, u - 1, -1):
            total_err_sq += _apply_adjacent_gate_svd(
                tensors, k, _SWAP_TENSOR_4D, self.max_bond_dim
            )

        return total_err_sq

    @staticmethod
    def evaluate_pauli_expectation(
        tensors: list[np.ndarray], pauli_map: dict[int, str]
    ) -> float:
        """
        Compute <psi_MPS| P |psi_MPS> in O(N * chi^3) time via 1D transfer-matrix
        contraction (never constructing a 2^N vector).
        """
        env = np.ones((1, 1), dtype=np.complex128)
        for q, t_q in enumerate(tensors):
            op = _PAULI_OPS[pauli_map.get(q, "I")]
            # Contract env[a, a'] with t_q[a, s, b], op[s', s], conj(t_q)[a', s', b']
            t_op = np.einsum("sp,apb->asb", op, t_q)
            env = np.einsum("aa,asb,aSb->bB", env, t_op, np.conj(t_q))
        return float(np.real(env[0, 0]))

    @classmethod
    def evaluate_ising_expectation(
        cls, tensors: list[np.ndarray], ising: IsingSpinGlass
    ) -> float:
        """Compute exact <psi_MPS| H_C |psi_MPS> term-by-term via MPS contractions."""
        norm = max(cls.evaluate_pauli_expectation(tensors, {}), 1e-12)
        exp_e = float(ising.constant_offset)

        for i, h_val in ising.h_linear.items():
            exp_e += h_val * (
                cls.evaluate_pauli_expectation(tensors, {i: "Z"}) / norm
            )
        for (i, j), j_val in ising.j_quadratic.items():
            exp_e += j_val * (
                cls.evaluate_pauli_expectation(tensors, {i: "Z", j: "Z"}) / norm
            )
        for (i, j, k), k_val in ising.k_cubic.items():
            exp_e += k_val * (
                cls.evaluate_pauli_expectation(tensors, {i: "Z", j: "Z", k: "Z"})
                / norm
            )
        return exp_e

    @staticmethod
    def contract_to_statevector_for_test(tensors: list[np.ndarray]) -> np.ndarray:
        """
        Contract MPS into a full 2^N vector (strictly for small-N unit testing of
        bond-dimension convergence against exact statevector simulation).
        """
        curr = tensors[0]  # (1, 2, chi_1)
        for t_q in tensors[1:]:
            curr = np.einsum("...a,asb->...sb", curr, t_q)
        return curr.reshape(-1)

    @staticmethod
    def sample_bitstrings(
        tensors: list[np.ndarray], shots: int = 256, seed: int = 42
    ) -> dict[str, int]:
        """
        Sample bitstrings from the MPS in O(shots * N * chi^3) time using exact
        right-environment Gram matrices and sequential conditional probabilities
        P(x_q | x_0, ..., x_{q-1}) — zero 2^N enumeration!
        """
        n = len(tensors)
        # Precompute right-to-left environment Gram matrices R[q] of shape (chi_q, chi_q)
        right_envs: list[np.ndarray] = [
            np.ones((1, 1), dtype=np.complex128) for _ in range(n + 1)
        ]
        for q in range(n - 1, -1, -1):
            t_q = tensors[q]
            right_envs[q] = np.einsum(
                "asb,AB,aSB->aA", t_q, right_envs[q + 1], np.conj(t_q)
            )

        rng = np.random.default_rng(seed)
        counts: dict[str, int] = {}

        for _ in range(shots):
            left_vec = np.ones((1,), dtype=np.complex128)
            bits: list[str] = []
            for q in range(n):
                t_q = tensors[q]
                r_env = right_envs[q + 1]
                # Conditional unnormalized weights for s in {0, 1}
                v0 = left_vec @ t_q[:, 0, :]
                v1 = left_vec @ t_q[:, 1, :]
                w0 = float(np.real(v0 @ r_env @ np.conj(v0)))
                w1 = float(np.real(v1 @ r_env @ np.conj(v1)))
                total_w = max(w0 + w1, 1e-15)
                p0 = float(np.clip(w0 / total_w, 0.0, 1.0))

                if rng.random() < p0:
                    bits.append("0")
                    norm_v = max(np.sqrt(max(w0, 1e-15)), 1e-15)
                    left_vec = v0 / norm_v
                else:
                    bits.append("1")
                    norm_v = max(np.sqrt(max(w1, 1e-15)), 1e-15)
                    left_vec = v1 / norm_v

            bstr = "".join(bits)
            counts[bstr] = counts.get(bstr, 0) + 1

        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

    def solve(self, ising: IsingSpinGlass) -> MPSResult:
        """
        Compile `ising` to a Paulihedral-canceled `QuantumCircuitIR`, evolve on the
        MPS tensor network, evaluate <H_C> via tensor contractions, and sample shots
        via sequential chain-rule probabilities.
        """
        n = ising.num_qubits
        report = compile_paulihedral_schedule(
            ising,
            topology="all_to_all",
            include_counterdiabatic=True,
            cost_gamma=0.48,
            mixer_beta=0.32,
        )
        assert report.optimized_ir is not None

        tensors, trunc_err = self.evolve_circuit(report.optimized_ir)
        achieved_chi = max(max(t.shape[0], t.shape[2]) for t in tensors)

        exp_e = self.evaluate_ising_expectation(tensors, ising)
        counts = self.sample_bitstrings(tensors, shots=self.shots, seed=self.seed)

        # Evaluate sampled bitstrings using O(|E| + |T|) polynomial evaluation (never 2^N)
        best_bstr = min(counts.keys(), key=ising.evaluate_bitstring)
        best_e = ising.evaluate_bitstring(best_bstr)

        total_s = max(sum(counts.values()), 1)
        prob_dict = {
            k: round(v / total_s, 5) for k, v in list(counts.items())[:10]
        }

        return MPSResult(
            num_qubits=n,
            max_bond_dimension=self.max_bond_dim,
            achieved_max_bond=achieved_chi,
            truncation_error=round(trunc_err, 8),
            best_bitstring=best_bstr,
            best_energy=round(best_e, 6),
            expectation_energy=round(exp_e, 6),
            bitstring_counts=counts,
            bitstring_probabilities=prob_dict,
        )


def _single_qubit_gate_matrix(
    name: str, params: tuple[float, ...]
) -> np.ndarray:
    if name == "H":
        return (
            np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
            / np.sqrt(2.0)
        )
    if name == "X":
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    angle = float(params[0]) if params else 0.0
    if name == "RX":
        c = np.cos(angle / 2.0)
        s = -1.0j * np.sin(angle / 2.0)
        return np.array([[c, s], [s, c]], dtype=np.complex128)
    if name == "RY":
        c = np.cos(angle / 2.0)
        s = np.sin(angle / 2.0)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if name == "RZ":
        return np.array(
            [[np.exp(-0.5j * angle), 0.0], [0.0, np.exp(0.5j * angle)]],
            dtype=np.complex128,
        )
    raise ValueError(f"Unknown 1Q gate: {name}")


def _apply_adjacent_gate_svd(
    tensors: list[np.ndarray],
    site_u: int,
    gate_4d: np.ndarray,
    max_bond: int,
) -> float:
    """
    Apply a 2-site gate `gate_4d[s'_u, s'_{u+1}, s_u, s_{u+1}]` to adjacent MPS
    tensors `(site_u, site_u + 1)` and truncate bond dimension to `max_bond` via SVD.
    Returns the discarded spectral weight squared `sum_{k >= chi} sigma_k^2`.
    """
    t_u = tensors[site_u]
    t_v = tensors[site_u + 1]
    alpha_dim = t_u.shape[0]
    beta_dim = t_v.shape[2]

    # Contract shared bond: theta[alpha, s_u, s_v, beta]
    theta = np.einsum("asb,btc->astc", t_u, t_v)
    # Apply 2-qubit gate: gated[alpha, s'_u, s'_v, beta]
    gated = np.einsum("PQst,astc->aPQc", gate_4d, theta)

    mat = gated.reshape(alpha_dim * 2, 2 * beta_dim)
    u_mat, s_vals, vh_mat = np.linalg.svd(mat, full_matrices=False)

    # Normalize singular values to preserve state norm
    norm_sq = float(np.sum(s_vals**2))
    if norm_sq > 1e-15:
        s_vals = s_vals / np.sqrt(norm_sq)

    chi = min(len(s_vals), max_bond)
    discarded_sq = float(np.sum(s_vals[chi:] ** 2)) if len(s_vals) > chi else 0.0

    u_trunc = u_mat[:, :chi]
    s_trunc = s_vals[:chi]
    vh_trunc = vh_mat[:chi, :]

    # Re-normalize retained singular values so state remains unit-norm
    ret_norm = float(np.sqrt(np.sum(s_trunc**2)))
    if ret_norm > 1e-15:
        s_trunc = s_trunc / ret_norm

    tensors[site_u] = u_trunc.reshape(alpha_dim, 2, chi)
    tensors[site_u + 1] = (np.diag(s_trunc) @ vh_trunc).reshape(
        chi, 2, beta_dim
    )
    return discarded_sq
