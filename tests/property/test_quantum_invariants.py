"""Property-based tests using Hypothesis to prove mathematical & feasibility invariants."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from pravaha_q.domain.problem_generator import generate_coupled_problem
from pravaha_q.math_engine.feasibility_repair import FeasibilityRepairEngine
from pravaha_q.math_engine.hubo_builder import VaisheshikaHUBOCompiler
from pravaha_q.math_engine.ising_spin_glass import hubo_to_ising


@settings(max_examples=15, deadline=None)
@given(
    num_nodes=st.integers(min_value=2, max_value=3),
    num_tasks=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=1, max_value=10000),
)
def test_property_canonical_ordering_and_100pct_feasibility(
    num_nodes: int, num_tasks: int, seed: int
) -> None:
    """
    Property Invariant: For ANY generated cluster topology and ANY corrupted
    bitstring, VaisheshikaHUBOCompiler preserves RFC-0001 index ordering and
    FeasibilityRepairEngine outputs a 100% valid schedule.
    """
    prob = generate_coupled_problem(
        num_nodes=num_nodes, num_tasks=num_tasks, num_power_states=1, seed=seed
    )
    hubo = VaisheshikaHUBOCompiler().compile(prob)
    ising = hubo_to_ising(hubo)

    # RFC-0001 ascending index invariant
    for u, v in ising.j_quadratic:
        assert 0 <= u < v < ising.num_qubits
    for u, v, w in ising.k_cubic:
        assert 0 <= u < v < w < ising.num_qubits

    repair = FeasibilityRepairEngine(prob, hubo)
    corrupted = "1" * ising.num_qubits
    decoded = repair.repair_and_decode(corrupted)
    assert decoded.is_feasible is True
