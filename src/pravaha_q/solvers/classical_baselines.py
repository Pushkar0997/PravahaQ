"""Classical exact eigensolver, simulated annealing, and greedy heuristic baselines."""

from __future__ import annotations

import time

import numpy as np

from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass
from pravaha_q.solvers.base import SolverResult


class ExactSpectralSolver:
    """Exact diagonal solver via full 2^N statevector enumeration."""

    def __init__(self) -> None:
        self.name = "ExactSpectral"

    def solve(self, ising: IsingSpinGlass) -> SolverResult:
        start_t = time.perf_counter()
        n = ising.num_qubits
        diag = ising.diagonal_vector()
        min_idx = int(np.argmin(diag))
        e_min = float(diag[min_idx])
        best_bstr = format(min_idx, f"0{n}b")

        exec_ms = (time.perf_counter() - start_t) * 1000.0

        return SolverResult(
            solver_name="ExactSpectral",
            sdk_name="NumPy/SciPy",
            num_qubits=n,
            best_bitstring=best_bstr,
            best_energy=round(e_min, 6),
            expectation_energy=round(e_min, 6),
            approximation_ratio=1.0,
            execution_time_ms=round(exec_ms, 2),
            bitstring_counts={best_bstr: 1024},
        )
