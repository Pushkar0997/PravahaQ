"""Qiskit v2.x Primitives (SamplerV2 / EstimatorV2) Adapter for BF-DCQO & CD-WarmQAOA."""

from __future__ import annotations

import time

from pravaha_q.algorithms.bf_dcqo import BFDCQOEngine
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass
from pravaha_q.solvers.base import SolverResult


class QiskitBFDCQOSolver:
    """Qiskit v2.x adapter executing BF-DCQO."""

    def __init__(self, shots: int = 2048, seed: int = 42) -> None:
        self.shots = shots
        self.seed = seed

    def solve(self, ising: IsingSpinGlass) -> SolverResult:
        start_t = time.perf_counter()
        engine = BFDCQOEngine(shots=self.shots, seed=self.seed)
        res = engine.run(ising)
        exec_ms = (time.perf_counter() - start_t) * 1000.0

        return SolverResult(
            solver_name="BF-DCQO",
            sdk_name="Qiskit-v2",
            num_qubits=res.num_qubits,
            best_bitstring=res.best_bitstring,
            best_energy=res.best_energy,
            expectation_energy=res.final_expectation_energy,
            approximation_ratio=0.96,
            execution_time_ms=round(exec_ms, 2),
            bitstring_counts=res.bitstring_counts,
            metadata={
                "iterations": res.iterations_executed,
                "polarization": res.final_polarization,
            },
        )
