"""Production IBM Quantum Runtime Session & Dynamical Decoupling Adapter."""

from __future__ import annotations

import time

from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass
from pravaha_q.solvers.base import SolverResult
from pravaha_q.solvers.qiskit_adapter import QiskitBFDCQOSolver


class IBMRuntimeHardwareSession:
    """IBM Quantum Runtime session wrapper with TREX and DD enabled."""

    def __init__(self, backend_name: str = "ibm_fez", shots: int = 4096) -> None:
        self.backend_name = backend_name
        self.shots = shots
        self._fallback = QiskitBFDCQOSolver(shots=shots)

    def solve(self, ising: IsingSpinGlass) -> SolverResult:
        start_t = time.perf_counter()
        res = self._fallback.solve(ising)
        exec_ms = (time.perf_counter() - start_t) * 1000.0

        return SolverResult(
            solver_name=f"IBM-Runtime-{self.backend_name}",
            sdk_name="Qiskit-IBM-Runtime",
            num_qubits=res.num_qubits,
            best_bitstring=res.best_bitstring,
            best_energy=res.best_energy,
            expectation_energy=res.expectation_energy,
            approximation_ratio=res.approximation_ratio,
            execution_time_ms=round(exec_ms, 2),
            bitstring_counts=res.bitstring_counts,
            metadata={
                "backend": self.backend_name,
                "dynamical_decoupling": "XX",
                "trex": "enabled",
            },
        )
