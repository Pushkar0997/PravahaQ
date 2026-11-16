"""Unified SolverStrategy protocol and standardized SolverResult telemetry schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass


@dataclass(frozen=True)
class SolverResult:
    """Standardized multi-SDK solver telemetry response."""

    solver_name: str
    sdk_name: str
    num_qubits: int
    best_bitstring: str
    best_energy: float
    expectation_energy: float
    approximation_ratio: float
    execution_time_ms: float
    bitstring_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, str | float] = field(default_factory=dict)


class SolverStrategy(Protocol):
    """Protocol defining the interface for all PravahaQ quantum and classical solvers."""

    def solve(self, ising: IsingSpinGlass) -> SolverResult: ...
