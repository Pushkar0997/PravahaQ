"""PennyLane Differentiable QNode Adapter for Multistage Pauli Correlation Encoding."""

from __future__ import annotations

import time

from pravaha_q.encodings.pauli_correlation_encoding import MultistagePCEEncoder
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass
from pravaha_q.math_engine.validator import SolverStatus
from pravaha_q.solvers.base import ExecutionMode, SolverResult


class PennyLanePCESolver:
    """PennyLane adapter executing Multistage Pauli Correlation Encoding."""

    def __init__(
        self,
        num_qubits: int = 6,
        seed: int = 42,
        execution_mode: ExecutionMode = ExecutionMode.IDEAL_STATEVECTOR,
    ) -> None:
        self.num_qubits = num_qubits
        self.seed = seed
        self.execution_mode = execution_mode

    def solve(self, ising: IsingSpinGlass) -> SolverResult:
        start_t = time.perf_counter()
        q_used = min(self.num_qubits, ising.num_qubits)
        encoder = MultistagePCEEncoder(num_qubits=q_used, seed=self.seed)
        res = encoder.solve(ising)
        exec_ms = (time.perf_counter() - start_t) * 1000.0

        counts = {res.best_bitstring: 1024}
        return SolverResult(
            solver_name="Multistage-PCE",
            sdk_name="PennyLane",
            num_qubits=ising.num_qubits,
            best_bitstring=res.best_bitstring,
            best_energy=res.best_energy,
            expectation_energy=res.best_energy,
            approximation_ratio=None,
            execution_time_ms=round(exec_ms, 2),
            bitstring_counts=counts,
            execution_mode=self.execution_mode,
            solver_status=SolverStatus.FEASIBLE_INCUMBENT,
            metadata={
                "qubits_used": res.num_qubits_used,
                "compression_ratio": res.compression_ratio,
            },
        )
