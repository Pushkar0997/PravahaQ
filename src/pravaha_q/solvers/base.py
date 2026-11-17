"""Unified SolverStrategy protocol and honest SolverResult telemetry schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass
from pravaha_q.math_engine.validator import ScheduleValidity, SolverStatus


class ExecutionMode(str, Enum):
    """Explicit separation between ideal statevectors, shot sampling, noise, and hardware."""

    CLASSICAL_REFERENCE = "CLASSICAL_REFERENCE"
    IDEAL_STATEVECTOR = "IDEAL_STATEVECTOR"
    SHOT_SIMULATION = "SHOT_SIMULATION"
    NOISY_SIMULATION = "NOISY_SIMULATION"
    HARDWARE = "HARDWARE"


@dataclass(frozen=True)
class SolverResult:
    """
    Standardized solver telemetry response.
    Strictly separates:
      - `best_energy`: Penalized surrogate energy E(x) = F(x) + penalties.
      - `operational_cost_f`: True operational objective F(x) on validated schedules.
      - `approximation_ratio`: Computed ONLY when a verified exact bound is known (else None).
      - `validity` & `solver_status`: Independent validation vs solver optimality claim.
    """

    solver_name: str
    sdk_name: str
    num_qubits: int
    best_bitstring: str
    best_energy: float
    expectation_energy: float
    approximation_ratio: float | None = None
    execution_time_ms: float = 0.0
    bitstring_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)
    execution_mode: ExecutionMode = ExecutionMode.IDEAL_STATEVECTOR
    operational_cost_f: float | None = None
    validity: ScheduleValidity = ScheduleValidity.VALID
    solver_status: SolverStatus = SolverStatus.UNKNOWN


class SolverStrategy(Protocol):
    """Protocol defining the interface for all PravahaQ quantum and classical solvers."""

    def solve(self, ising: IsingSpinGlass) -> SolverResult: ...
