"""Unit tests for Multistage PCE Qubit Compression, IQO-BnB, and Feasibility Repair."""

from __future__ import annotations

from pravaha_q.algorithms.iqo_branch_and_bound import IQOBranchAndBoundSolver
from pravaha_q.domain.problem_generator import (
    CoupledInfrastructureProblem,
    generate_coupled_problem,
)
from pravaha_q.encodings.pauli_correlation_encoding import MultistagePCEEncoder
from pravaha_q.math_engine.feasibility_repair import FeasibilityRepairEngine
from pravaha_q.math_engine.hubo_builder import (
    HUBOPolynomial,
    VaisheshikaHUBOCompiler,
)
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass, hubo_to_ising


def test_multistage_pce_sublinear_qubit_compression() -> None:
    """Verify compressing an 18-variable problem onto just 4 physical qubits."""
    prob = generate_coupled_problem(
        num_nodes=3, num_tasks=4, num_power_states=2, seed=42
    )
    assert prob.total_binary_variables == 18
    hubo = VaisheshikaHUBOCompiler().compile(prob)
    ising = hubo_to_ising(hubo)

    encoder = MultistagePCEEncoder(num_qubits=4, ansatz_layers=2, seed=42)
    res = encoder.solve(ising)
    assert res.num_variables == 18
    assert res.num_qubits_used == 4
    assert res.compression_ratio == 4.5
    assert len(res.best_bitstring) == 18


def test_iqo_branch_and_bound_and_feasibility_repair(
    small_coupled_problem: CoupledInfrastructureProblem,
    small_hubo: HUBOPolynomial,
    small_ising: IsingSpinGlass,
) -> None:
    iqo = IQOBranchAndBoundSolver(
        classical_cutoff_vars=3, max_backtrack_depth=2, seed=42
    )
    res = iqo.solve(small_ising)
    assert res.branches_explored >= 2
    assert len(res.best_bitstring) == small_ising.num_qubits

    repair = FeasibilityRepairEngine(small_coupled_problem, small_hubo)
    # Test repairing an intentionally invalid all-zero bitstring
    bad_bits = "0" * small_ising.num_qubits
    assert not repair.is_bitstring_feasible(bad_bits)
    decoded = repair.repair_and_decode(bad_bits)
    assert decoded.is_feasible is True
    assert decoded.was_repaired is True
