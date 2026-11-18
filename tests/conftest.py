"""Shared deterministic fixtures for the PravahaQ 4-tier verification suite."""

from __future__ import annotations

import pytest

from pravaha_q.domain.problem_generator import (
    CoupledInfrastructureProblem,
    generate_coupled_problem,
)
from pravaha_q.math_engine.hubo_builder import (
    HUBOPolynomial,
    VaisheshikaHUBOCompiler,
)
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass, hubo_to_ising


@pytest.fixture
def small_coupled_problem() -> CoupledInfrastructureProblem:
    """Return a deterministic 2-node, 2-task, 2-power-state instance (8 binary vars)."""
    return generate_coupled_problem(
        num_nodes=2, num_tasks=2, num_power_states=2, seed=42
    )


@pytest.fixture
def small_hubo(small_coupled_problem: CoupledInfrastructureProblem) -> HUBOPolynomial:
    """Return compiled HUBOPolynomial for small_coupled_problem."""
    return VaisheshikaHUBOCompiler(enable_cubic_terms=True).compile(
        small_coupled_problem
    )


@pytest.fixture
def small_ising(small_hubo: HUBOPolynomial) -> IsingSpinGlass:
    """Return exact IsingSpinGlass mapped from small_hubo."""
    return hubo_to_ising(small_hubo)
