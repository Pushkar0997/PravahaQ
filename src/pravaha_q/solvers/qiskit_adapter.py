"""
Qiskit v2.x Adapter (Driven by Canonical QuantumCircuitIR & Strict ExecutionMode)
=================================================================================
- Translates `QuantumCircuitIR` directly into a `qiskit.QuantumCircuit` when Qiskit
  is installed.
- Respects `ExecutionMode.SHOT_SIMULATION` vs `ExecutionMode.IDEAL_STATEVECTOR`.
- Never fabricates hardcoded `approximation_ratio` values (`None` unless verified).
"""

from __future__ import annotations

import time
from typing import Any

from pravaha_q.algorithms.bf_dcqo import BFDCQOEngine
from pravaha_q.compiler.circuit_ir import GateOp, QuantumCircuitIR
from pravaha_q.compiler.paulihedral_router import compile_paulihedral_schedule
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass
from pravaha_q.math_engine.validator import SolverStatus
from pravaha_q.solvers.base import ExecutionMode, SolverResult


def circuit_ir_to_qiskit(circuit: QuantumCircuitIR) -> Any:
    """Convert a canonical `QuantumCircuitIR` into a `qiskit.QuantumCircuit` if installed."""
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        return None

    qc = QuantumCircuit(circuit.num_qubits, circuit.num_qubits)
    for op in circuit.ops:
        g = op.name
        if g == "H":
            qc.h(op.qubits[0])
        elif g == "X":
            qc.x(op.qubits[0])
        elif g == "RX":
            qc.rx(float(op.params[0]), op.qubits[0])
        elif g == "RY":
            qc.ry(float(op.params[0]), op.qubits[0])
        elif g == "RZ":
            qc.rz(float(op.params[0]), op.qubits[0])
        elif g == "CX":
            qc.cx(op.qubits[0], op.qubits[1])
        elif g == "SWAP":
            qc.swap(op.qubits[0], op.qubits[1])
    return qc


class QiskitBFDCQOSolver:
    """Qiskit v2.x adapter executing BF-DCQO with explicit ExecutionMode separation."""

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
        use_shots = self.execution_mode in (
            ExecutionMode.SHOT_SIMULATION,
            ExecutionMode.NOISY_SIMULATION,
            ExecutionMode.HARDWARE,
        )
        engine = BFDCQOEngine(
            shots=self.shots,
            seed=self.seed,
            use_shot_estimates=use_shots,
        )
        res = engine.run(ising)
        pauli_rep = compile_paulihedral_schedule(ising, topology="heavy_hex")
        assert pauli_rep.optimized_ir is not None
        qk_circ = circuit_ir_to_qiskit(pauli_rep.optimized_ir)

        exec_ms = (time.perf_counter() - start_t) * 1000.0
        return SolverResult(
            solver_name="BF-DCQO",
            sdk_name="Qiskit-v2",
            num_qubits=res.num_qubits,
            best_bitstring=res.best_bitstring,
            best_energy=res.best_energy,
            expectation_energy=res.final_expectation_energy,
            approximation_ratio=None,
            execution_time_ms=round(exec_ms, 2),
            bitstring_counts=res.bitstring_counts,
            execution_mode=self.execution_mode,
            solver_status=SolverStatus.FEASIBLE_INCUMBENT,
            metadata={
                "iterations": res.iterations_executed,
                "polarization": res.final_polarization,
                "ir_2q_gates": pauli_rep.post_cancellation_cx_count,
                "routed_2q_gates": pauli_rep.total_routed_2q_gates,
                "qiskit_native_built": qk_circ is not None,
            },
        )
