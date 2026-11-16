"""PravahaQ Developer Python SDK Client."""

from __future__ import annotations

from pravaha_q.domain.problem_generator import generate_coupled_problem
from pravaha_q.math_engine.hubo_builder import VaisheshikaHUBOCompiler
from pravaha_q.math_engine.ising_spin_glass import hubo_to_ising
from pravaha_q.solvers.base import SolverResult
from pravaha_q.solvers.qiskit_adapter import QiskitBFDCQOSolver


class PravahaClient:
    """High-level Python SDK client for PravahaQ."""

    def __init__(self, default_sdk: str = "qiskit") -> None:
        self.default_sdk = default_sdk

    def optimize_cluster(self, num_nodes: int = 3, num_tasks: int = 4) -> SolverResult:
        prob = generate_coupled_problem(num_nodes=num_nodes, num_tasks=num_tasks)
        hubo = VaisheshikaHUBOCompiler().compile(prob)
        ising = hubo_to_ising(hubo)
        solver = QiskitBFDCQOSolver()
        return solver.solve(ising)
