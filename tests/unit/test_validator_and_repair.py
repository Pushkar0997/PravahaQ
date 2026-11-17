"""
Unit tests verifying Repair Batch R1 (Steps 1, 2, and 3):
  1. Elimination of y_{n,0} variables when num_power_states == 1 (R == 1).
  2. Independent ScheduleValidator catches capacity and deadline violations.
  3. Overloaded / infeasible instances return `is_feasible=False` and `PROVEN_INFEASIBLE`
     (repair NEVER manufactures a fake success flag).
  4. Minimum-Hamming MILP repair finds the closest valid schedule when feasible.
  5. StructuredEnumerationSolver finds the exact minimum operational cost F(x*).
"""

from __future__ import annotations

from pravaha_q.domain.problem_generator import generate_coupled_problem
from pravaha_q.math_engine.feasibility_repair import FeasibilityRepairEngine
from pravaha_q.math_engine.hubo_builder import VaisheshikaHUBOCompiler
from pravaha_q.math_engine.validator import (
    ScheduleValidator,
    ScheduleValidity,
    SolverStatus,
)
from pravaha_q.solvers.classical_baselines import (
    GreedyOperationalSolver,
    StructuredEnumerationSolver,
)


def test_r_equals_1_eliminates_redundant_power_mode_variables() -> None:
    """When R == 1, y_{n,0}=1 must be substituted out so only M * N_node variables exist."""
    prob_r1 = generate_coupled_problem(num_nodes=2, num_tasks=3, num_power_states=1, seed=42)
    assert prob_r1.has_mode_variables is False
    assert prob_r1.num_power_vars == 0
    assert prob_r1.total_binary_variables == 6  # 3 tasks * 2 nodes (no unconstrained y vars!)

    prob_r2 = generate_coupled_problem(num_nodes=2, num_tasks=3, num_power_states=2, seed=42)
    assert prob_r2.has_mode_variables is True
    assert prob_r2.num_power_vars == 4
    assert prob_r2.total_binary_variables == 10


def test_independent_validator_rejects_one_hot_schedule_that_violates_capacity() -> None:
    """A one-hot assignment that overloads a single node must be flagged INVALID."""
    prob = generate_coupled_problem(num_nodes=2, num_tasks=3, num_power_states=1, seed=42)
    validator = ScheduleValidator(prob)

    # Put all 3 tasks on Node 0: x_{0,0}=1, x_{1,0}=1, x_{2,0}=1 -> "101010"
    overloaded_one_hot = "101010"
    report = validator.validate(overloaded_one_hot)
    assert report.is_valid is False
    assert report.validity == ScheduleValidity.INVALID
    assert any(
        v.constraint_type == "NODE_CAPACITY_ENVELOPE_WATTS" for v in report.violations
    )


def test_overloaded_infeasible_instance_never_manufactures_success_flag() -> None:
    """
    When total task demand exceeds total cluster capacity (`capacity_scale=0.15`),
    both MILP repair and StructuredEnumerationSolver must report `PROVEN_INFEASIBLE`
    and `is_feasible=False`.
    """
    impossible_prob = generate_coupled_problem(
        num_nodes=2,
        num_tasks=3,
        num_power_states=1,
        seed=42,
        capacity_scale=0.15,
    )
    hubo = VaisheshikaHUBOCompiler().compile(impossible_prob)
    repair = FeasibilityRepairEngine(impossible_prob, hubo)

    decoded = repair.repair_and_decode("100110")
    assert decoded.is_feasible is False
    assert decoded.validity == ScheduleValidity.INVALID
    assert decoded.solver_status == SolverStatus.PROVEN_INFEASIBLE

    enum_res = StructuredEnumerationSolver().solve_problem(impossible_prob, hubo)
    assert enum_res.validity == ScheduleValidity.INVALID
    assert enum_res.solver_status == SolverStatus.PROVEN_INFEASIBLE


def test_milp_minimum_hamming_repair_and_structured_reference() -> None:
    """Verify minimum-Hamming MILP repair and exact structured enumeration on a feasible problem."""
    prob = generate_coupled_problem(num_nodes=2, num_tasks=3, num_power_states=2, seed=42)
    hubo = VaisheshikaHUBOCompiler().compile(prob)

    enum_res = StructuredEnumerationSolver().solve_problem(prob, hubo)
    assert enum_res.validity == ScheduleValidity.VALID
    assert enum_res.solver_status == SolverStatus.OPTIMAL
    assert enum_res.operational_cost_f is not None

    greedy_res = GreedyOperationalSolver().solve_problem(prob, hubo)
    assert greedy_res.validity == ScheduleValidity.VALID
    assert greedy_res.operational_cost_f >= enum_res.operational_cost_f - 1e-6

    # Flip 1 bit from the optimal schedule and verify MILP repairs it with Hamming distance == 1
    opt_bits = [int(c) for c in enum_res.best_bitstring]
    corrupted = list(opt_bits)
    corrupted[0] ^= 1
    decoded = FeasibilityRepairEngine(prob, hubo).repair_and_decode(corrupted)
    assert decoded.is_feasible is True
    assert decoded.hamming_distance == 1
