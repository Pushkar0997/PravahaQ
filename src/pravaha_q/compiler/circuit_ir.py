"""
Single Canonical Quantum Circuit Intermediate Representation (QuantumCircuitIR)
===============================================================================
Provides one authoritative circuit definition used across:
  1. Ideal and noisy statevector simulation
  2. Matrix Product State (MPS) tensor-network contraction
  3. Algebraic gate cancellation and hardware SWAP routing
  4. OpenQASM 3.0 and NVIDIA CUDA-Q code generation
  5. Qiskit, PennyLane, and Google Cirq SDK translation
  6. Unitary gate folding U -> U (U^dagger U)^{(c-1)/2} for Zero-Noise Extrapolation
"""

from __future__ import annotations

from dataclasses import dataclass, field
import networkx as nx
import numpy as np

from pravaha_q.algorithms.agp_commutator import (
    CounterdiabaticImpulseSchedule,
    compute_analytical_agp_schedule,
)
from pravaha_q.algorithms.bf_dcqo import (
    apply_single_qubit_rx,
    apply_single_qubit_ry,
    apply_single_qubit_rz,
)
from pravaha_q.compiler.hardware_graphs import HardwareTopologyGraph
from pravaha_q.encodings.pauli_correlation_encoding import _apply_cnot_tensor
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass


@dataclass(frozen=True)
class GateOp:
    """
    Immutable elementary quantum gate operation in the canonical IR.
    Supported gate names:
      - 1-Qubit: "H", "X", "RX", "RY", "RZ"
      - 2-Qubit: "CX", "SWAP"
    """

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = field(default_factory=tuple)

    def inverse(self) -> GateOp:
        """Return the exact Hermitian adjoint (U^dagger) of this gate."""
        if self.name in ("H", "X", "CX", "SWAP"):
            return self
        if self.name in ("RX", "RY", "RZ"):
            return GateOp(
                name=self.name,
                qubits=self.qubits,
                params=(-float(self.params[0]),),
            )
        raise ValueError(f"Unsupported gate name for inversion: {self.name}")


@dataclass(frozen=True)
class QuantumCircuitIR:
    """Canonical circuit IR shared by simulators, compilers, and SDK adapters."""

    num_qubits: int
    ops: tuple[GateOp, ...] = field(default_factory=tuple)
    measured_qubits: tuple[int, ...] = field(default_factory=tuple)
    logical_to_physical: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.measured_qubits:
            object.__setattr__(
                self, "measured_qubits", tuple(range(self.num_qubits))
            )
        if not self.logical_to_physical:
            object.__setattr__(
                self, "logical_to_physical", tuple(range(self.num_qubits))
            )

    def count_1q_gates(self) -> int:
        return sum(1 for op in self.ops if len(op.qubits) == 1)

    def count_2q_gates(self) -> int:
        return sum(
            3 if op.name == "SWAP" else 1
            for op in self.ops
            if len(op.qubits) == 2
        )

    def depth(self) -> int:
        """Compute exact circuit layer depth across all qubits."""
        if not self.ops:
            return 0
        qubit_layer = [0] * self.num_qubits
        for op in self.ops:
            max_prev = max(qubit_layer[q] for q in op.qubits)
            step = 3 if op.name == "SWAP" else 1
            for q in op.qubits:
                qubit_layer[q] = max_prev + step
        return max(qubit_layer, default=0)

    def dagger(self) -> QuantumCircuitIR:
        """Return the exact adjoint circuit U^dagger (reversed inverted gates)."""
        inv_ops = tuple(op.inverse() for op in reversed(self.ops))
        return QuantumCircuitIR(
            num_qubits=self.num_qubits,
            ops=inv_ops,
            measured_qubits=self.measured_qubits,
            logical_to_physical=self.logical_to_physical,
        )

    def fold_unitary(self, scale_factor: int = 1) -> QuantumCircuitIR:
        """
        Construct the unitary-folded circuit U (U^dagger U)^k where
        scale_factor = 2*k + 1 in {1, 3, 5, ...} for Zero-Noise Extrapolation.
        In the noiseless limit, fold_unitary(scale_factor) is unitarily identical to U.
        """
        if scale_factor < 1 or scale_factor % 2 == 0:
            raise ValueError(
                f"Unitary folding scale_factor must be a positive odd integer (got {scale_factor})."
            )
        num_folds = (scale_factor - 1) // 2
        if num_folds == 0:
            return self
        u_dag_ops = tuple(op.inverse() for op in reversed(self.ops))
        folded_ops = list(self.ops)
        for _ in range(num_folds):
            folded_ops.extend(u_dag_ops)
            folded_ops.extend(self.ops)
        return QuantumCircuitIR(
            num_qubits=self.num_qubits,
            ops=tuple(folded_ops),
            measured_qubits=self.measured_qubits,
            logical_to_physical=self.logical_to_physical,
        )

    def simulate_statevector(self, per_gate_depol_error: float = 0.0) -> np.ndarray:
        """
        Execute `self.ops` sequentially from |0>^{otimes n} to produce the exact
        2^n statevector. If `per_gate_depol_error > 0`, applies an effective
        circuit-level depolarizing damping factor to the final probability distribution.
        """
        n = self.num_qubits
        state = np.zeros(1 << n, dtype=np.complex128)
        state[0] = 1.0 + 0.0j
        psi = state.reshape((2,) * n)

        for op in self.ops:
            g = op.name
            if g == "H":
                q = op.qubits[0]
                h_mat = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
                psi = np.tensordot(h_mat, psi, axes=([1], [q]))
                psi = np.moveaxis(psi, 0, q)
            elif g == "X":
                q = op.qubits[0]
                psi = apply_single_qubit_rx(psi, q, float(np.pi)) * 1.0j
            elif g == "RX":
                psi = apply_single_qubit_rx(psi, op.qubits[0], float(op.params[0]))
            elif g == "RY":
                psi = apply_single_qubit_ry(psi, op.qubits[0], float(op.params[0]))
            elif g == "RZ":
                psi = apply_single_qubit_rz(psi, op.qubits[0], float(op.params[0]))
            elif g == "CX":
                psi = _apply_cnot_tensor(psi, op.qubits[0], op.qubits[1], n)
            elif g == "SWAP":
                u, v = op.qubits
                psi = _apply_cnot_tensor(psi, u, v, n)
                psi = _apply_cnot_tensor(psi, v, u, n)
                psi = _apply_cnot_tensor(psi, u, v, n)
            else:
                raise ValueError(f"Unsupported gate in QuantumCircuitIR: {g}")

        vec = psi.reshape(-1)
        if per_gate_depol_error > 0.0:
            # Effective survival probability p_surv = (1 - eps_2q)^{N_2q} * (1 - 0.1*eps_2q)^{N_1q}
            n_2q = self.count_2q_gates()
            n_1q = self.count_1q_gates()
            fidelity = ((1.0 - per_gate_depol_error) ** n_2q) * (
                (1.0 - 0.1 * per_gate_depol_error) ** n_1q
            )
            probs = np.abs(vec) ** 2
            dim = 1 << n
            noisy_probs = fidelity * probs + (1.0 - fidelity) * (1.0 / dim)
            phases = np.angle(vec)
            vec = np.sqrt(np.clip(noisy_probs, 0.0, 1.0)) * np.exp(1.0j * phases)

        return vec

    def route_to_topology(
        self, topology: HardwareTopologyGraph
    ) -> tuple[QuantumCircuitIR, int]:
        """
        Route all 2-qubit gates onto `topology` by inserting explicit `CX`-decomposed
        SWAP chains along shortest paths and updating the logical-to-physical mapping.
        Returns `(routed_circuit_ir, swap_gates_inserted)`.
        """
        if topology.topology_type == "all_to_all":
            return self, 0

        g = topology.to_networkx()
        # Ensure graph covers all qubits in the circuit
        for q in range(self.num_qubits):
            if not g.has_node(q):
                g.add_node(q)
                if q > 0:
                    g.add_edge(q - 1, q)

        log_to_phys = list(range(self.num_qubits))
        phys_to_log = list(range(self.num_qubits))
        routed_ops: list[GateOp] = []
        swaps_inserted = 0

        def swap_physical(p_a: int, p_b: int) -> None:
            nonlocal swaps_inserted
            # Decompose SWAP(p_a, p_b) into 3 native CX gates
            routed_ops.append(GateOp("CX", (p_a, p_b)))
            routed_ops.append(GateOp("CX", (p_b, p_a)))
            routed_ops.append(GateOp("CX", (p_a, p_b)))
            swaps_inserted += 1
            l_a, l_b = phys_to_log[p_a], phys_to_log[p_b]
            phys_to_log[p_a], phys_to_log[p_b] = l_b, l_a
            log_to_phys[l_a], log_to_phys[l_b] = p_b, p_a

        for op in self.ops:
            if len(op.qubits) == 1:
                p_q = log_to_phys[op.qubits[0]]
                routed_ops.append(GateOp(op.name, (p_q,), op.params))
            elif len(op.qubits) == 2:
                l_u, l_v = op.qubits
                p_u, p_v = log_to_phys[l_u], log_to_phys[l_v]
                if not g.has_edge(p_u, p_v):
                    try:
                        path = nx.shortest_path(g, source=p_u, target=p_v)
                    except nx.NetworkXNoPath:
                        path = [p_u, p_v]
                    # Swap l_u along path until adjacent to p_v
                    for step_idx in range(len(path) - 2):
                        swap_physical(path[step_idx], path[step_idx + 1])
                    p_u = log_to_phys[l_u]
                    p_v = log_to_phys[l_v]
                routed_ops.append(GateOp(op.name, (p_u, p_v), op.params))

        return (
            QuantumCircuitIR(
                num_qubits=self.num_qubits,
                ops=tuple(routed_ops),
                measured_qubits=tuple(log_to_phys[q] for q in range(self.num_qubits)),
                logical_to_physical=tuple(log_to_phys),
            ),
            swaps_inserted,
        )


def build_bf_dcqo_circuit_ir(
    ising: IsingSpinGlass,
    bias_fields: dict[int, float] | None = None,
    agp_schedule: CounterdiabaticImpulseSchedule | None = None,
    cost_gamma: float = 0.45,
    mixer_beta: float = 0.30,
    include_counterdiabatic: bool = True,
    pauli_term_order: list[tuple[tuple[int, ...], float]] | None = None,
) -> QuantumCircuitIR:
    """
    Construct the canonical QuantumCircuitIR for a single BF-DCQO / QAOA impulse step.
    Used identically by the statevector simulator, MPS simulator, OpenQASM 3.0 exporter,
    and SDK adapters.
    """
    n = ising.num_qubits
    h_norm, j_norm, k_norm = ising.normalized_coefficients()
    scale = max(ising.spectral_scale, 1e-9)
    agp = agp_schedule or compute_analytical_agp_schedule(
        ising, bias_fields=bias_fields
    )

    ops: list[GateOp] = []

    # 1. Uniform superposition |+>^{otimes n}
    for q in range(n):
        ops.append(GateOp("H", (q,)))

    # 2. Longitudinal bias-field initial tilt Ry(theta_bias_i)
    if bias_fields:
        for i, b_val in sorted(bias_fields.items()):
            if abs(b_val) > 1e-12:
                tilt = float(-0.65 * np.arctan(b_val / scale))
                ops.append(GateOp("RY", (i,), (tilt,)))

    # 3. Counterdiabatic AGP Impulse (1-body Y, 2-body YZ+ZY, 3-body YZZ+ZYZ+ZZY)
    if include_counterdiabatic:
        for i, y_ang in sorted(agp.y_1body_angles.items()):
            if abs(y_ang) > 1e-12:
                ops.append(GateOp("RY", (i,), (2.0 * float(y_ang),)))

        for (u, v), yz_ang in sorted(agp.yz_2body_angles.items()):
            if abs(yz_ang) > 1e-12:
                ang = 2.0 * float(yz_ang)
                # exp(-i yz_ang Y_u Z_v)
                ops.append(GateOp("RX", (u,), (float(np.pi / 2.0),)))
                ops.append(GateOp("CX", (u, v)))
                ops.append(GateOp("RZ", (v,), (ang,)))
                ops.append(GateOp("CX", (u, v)))
                ops.append(GateOp("RX", (u,), (float(-np.pi / 2.0),)))
                # exp(-i yz_ang Z_u Y_v)
                ops.append(GateOp("RX", (v,), (float(np.pi / 2.0),)))
                ops.append(GateOp("CX", (u, v)))
                ops.append(GateOp("RZ", (v,), (ang,)))
                ops.append(GateOp("CX", (u, v)))
                ops.append(GateOp("RX", (v,), (float(-np.pi / 2.0),)))

        for (u, v, w), yzz_ang in sorted(agp.yzz_3body_angles.items()):
            if abs(yzz_ang) > 1e-12:
                ang = 2.0 * float(yzz_ang)
                # Apply all 3 cyclic permutations: Y_u Z_v Z_w, Z_u Y_v Z_w, Z_u Z_v Y_w
                for target_y in (u, v, w):
                    ops.append(GateOp("RX", (target_y,), (float(np.pi / 2.0),)))
                    ops.append(GateOp("CX", (u, v)))
                    ops.append(GateOp("CX", (v, w)))
                    ops.append(GateOp("RZ", (w,), (ang,)))
                    ops.append(GateOp("CX", (v, w)))
                    ops.append(GateOp("CX", (u, v)))
                    ops.append(GateOp("RX", (target_y,), (float(-np.pi / 2.0),)))

    # 4. Cost Hamiltonian Phase Separator exp(-i cost_gamma H_C)
    for i, h_val in sorted(h_norm.items()):
        if abs(h_val) > 1e-12:
            ops.append(GateOp("RZ", (i,), (2.0 * cost_gamma * float(h_val),)))

    if pauli_term_order is None:
        multi_terms: list[tuple[tuple[int, ...], float]] = []
        for pair, val in sorted(ising.j_quadratic.items()):
            multi_terms.append((pair, float(val)))
        for triad, val in sorted(ising.k_cubic.items()):
            multi_terms.append((triad, float(val)))
    else:
        multi_terms = pauli_term_order

    for supp, raw_val in multi_terms:
        norm_val = raw_val / scale
        angle = 2.0 * cost_gamma * norm_val
        if abs(angle) <= 1e-12:
            continue
        if len(supp) == 2:
            u, v = supp
            ops.append(GateOp("CX", (u, v)))
            ops.append(GateOp("RZ", (v,), (angle,)))
            ops.append(GateOp("CX", (u, v)))
        elif len(supp) == 3:
            u, v, w = supp
            ops.append(GateOp("CX", (u, v)))
            ops.append(GateOp("CX", (v, w)))
            ops.append(GateOp("RZ", (w,), (angle,)))
            ops.append(GateOp("CX", (v, w)))
            ops.append(GateOp("CX", (u, v)))

    # 5. Transverse Field Mixer Layer exp(i mixer_beta sum_i X_i) == RX(-2 * mixer_beta)
    if abs(mixer_beta) > 1e-12:
        for q in range(n):
            ops.append(GateOp("RX", (q,), (-2.0 * float(mixer_beta),)))

    return QuantumCircuitIR(num_qubits=n, ops=tuple(ops))
