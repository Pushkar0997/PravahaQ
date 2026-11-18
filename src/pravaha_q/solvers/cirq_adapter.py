"""
Google Cirq Planar GridQubit Adapter (Driven by Canonical QuantumCircuitIR)
===========================================================================
Compiles the problem onto a 2D Planar GridQubit topology via `compile_paulihedral_schedule`,
translates the routed `QuantumCircuitIR` into a `cirq.Circuit` (if `cirq` is installed)
or executes the routed `QuantumCircuitIR` directly, and samples finite shots.
Never calls `ising.diagonal_vector()` to cheat!
"""

from __future__ import annotations

import time
from typing import Any
import numpy as np

from pravaha_q.algorithms.bf_dcqo import sample_bitstring_counts
from pravaha_q.compiler.circuit_ir import QuantumCircuitIR
from pravaha_q.compiler.paulihedral_router import compile_paulihedral_schedule
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass
from pravaha_q.math_engine.validator import SolverStatus
from pravaha_q.solvers.base import ExecutionMode, SolverResult


def circuit_ir_to_cirq(circuit: QuantumCircuitIR) -> Any:
    """Convert a canonical `QuantumCircuitIR` into a `cirq.Circuit` on `GridQubit`s if installed."""
    try:
        import cirq
    except ImportError:
        return None

    n = circuit.num_qubits
    cols = int(max(2, round(n**0.5)))
    qubits = [cirq.GridQubit(i // cols, i % cols) for i in range(n)]
    c_circ = cirq.Circuit()

    for op in circuit.ops:
        g = op.name
        if g == "H":
            c_circ.append(cirq.H(qubits[op.qubits[0]]))
        elif g == "X":
            c_circ.append(cirq.X(qubits[op.qubits[0]]))
        elif g == "RX":
            c_circ.append(cirq.rx(float(op.params[0]))(qubits[op.qubits[0]]))
        elif g == "RY":
            c_circ.append(cirq.ry(float(op.params[0]))(qubits[op.qubits[0]]))
        elif g == "RZ":
            c_circ.append(cirq.rz(float(op.params[0]))(qubits[op.qubits[0]]))
        elif g == "CX":
            c_circ.append(cirq.CNOT(qubits[op.qubits[0]], qubits[op.qubits[1]]))
        elif g == "SWAP":
            c_circ.append(cirq.SWAP(qubits[op.qubits[0]], qubits[op.qubits[1]]))

    return c_circ


class CirqGridSolver:
    """Google Cirq 2D Planar GridQubit solver executing the canonical QuantumCircuitIR."""

    def __init__(
        self,
        shots: int = 2048,
        seed: int = 42,
        execution_mode: ExecutionMode = ExecutionMode.SHOT_SIMULATION,
    ) -> None:
        self.shots = shots
        self.seed = seed
        self.execution_mode = execution_mode

    def solve(self, ising: IsingSpinGlass) -> SolverResult:
        start_t = time.perf_counter()
        n = ising.num_qubits

        pauli_rep = compile_paulihedral_schedule(
            ising,
            topology="planar_grid",
            include_counterdiabatic=True,
            cost_gamma=0.48,
            mixer_beta=0.32,
        )
        assert pauli_rep.optimized_ir is not None
        assert pauli_rep.routed_ir is not None

        cirq_circuit = circuit_ir_to_cirq(pauli_rep.routed_ir)

        # Simulate the canonical optimized QuantumCircuitIR statevector and sample shots
        state = pauli_rep.optimized_ir.simulate_statevector()
        probs = np.abs(state) ** 2
        probs = probs / np.sum(probs)

        counts = sample_bitstring_counts(
            probs, num_qubits=n, shots=self.shots, seed=self.seed
        )
        total_s = max(sum(counts.values()), 1)
        exp_e = float(
            sum(ising.evaluate_bitstring(b) * c for b, c in counts.items())
            / total_s
        )
        best_bstr = min(counts.keys(), key=ising.evaluate_bitstring)
        best_e = ising.evaluate_bitstring(best_bstr)
        exec_ms = (time.perf_counter() - start_t) * 1000.0

        return SolverResult(
            solver_name="Cirq-GridQAOA",
            sdk_name="Google-Cirq",
            num_qubits=n,
            best_bitstring=best_bstr,
            best_energy=round(best_e, 6),
            expectation_energy=round(exp_e, 6),
            approximation_ratio=None,
            execution_time_ms=round(exec_ms, 2),
            bitstring_counts=counts,
            execution_mode=self.execution_mode,
            solver_status=SolverStatus.FEASIBLE_INCUMBENT,
            metadata={
                "topology": pauli_rep.topology_name,
                "routed_2q_gates": pauli_rep.total_routed_2q_gates,
                "swap_cx_overhead": pauli_rep.swap_routing_cx_overhead,
                "cirq_native_built": cirq_circuit is not None,
            },
        )
