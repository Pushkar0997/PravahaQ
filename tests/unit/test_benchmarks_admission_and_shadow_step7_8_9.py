"""
Unit tests verifying GPT-6 Astra Steps 7, 8, and 9:
  - Step 7: Honest recomputed ablation suite (`Random+Repair`, `Greedy+Repair`,
    `Separate+Repair`, `BF-DCQO+Repair`, `StructuredExact`, and overloaded rejection).
  - Step 8: Algorithm-specific resource/memory admission guard (`check_execution_admission`).
  - Step 9: Shadow-Mode cluster evaluation across normal and overloaded workload slices.
"""

from __future__ import annotations

import pytest

from pravaha_q.benchmarks.shadow_evaluator import run_shadow_mode_evaluation
from pravaha_q.benchmarks.suite_runner import run_full_ablation_suite
from pravaha_q.domain.problem_generator import generate_coupled_problem
from pravaha_q.infrastructure.admission_guard import (
    CompatibilityError,
    ResourceAdmissionError,
    ResourceBudget,
    check_execution_admission,
)


def test_step8_resource_admission_guard_and_compatibility() -> None:
    """Verify algorithm-specific qubit/memory ceilings and backend compatibility."""
    prob_small = generate_coupled_problem(
        num_nodes=2, num_tasks=3, num_power_states=1, seed=42
    )
    decision = check_execution_admission(prob_small, "bf_dcqo", "qiskit")
    assert decision.admitted is True
    assert decision.num_qubits_allocated == 6

    # Reject incompatible algorithm + backend pair
    with pytest.raises(CompatibilityError):
        check_execution_admission(prob_small, "bf_dcqo", "pennylane")

    # Reject dense statevector simulation when problem exceeds qubit budget (e.g. 24 vars > 18Q)
    prob_large = generate_coupled_problem(
        num_nodes=4, num_tasks=6, num_power_states=1, seed=42
    )
    assert prob_large.total_binary_variables == 24
    with pytest.raises(ResourceAdmissionError):
        check_execution_admission(prob_large, "bf_dcqo", "qiskit")

    # However, compressed PCE (on 5 qubits) and MPS (chi=16) MUST be admitted on the same 24-var problem!
    pce_dec = check_execution_admission(
        prob_large, "pce", "pennylane", pce_qubits=5
    )
    assert pce_dec.admitted is True
    assert pce_dec.num_qubits_allocated == 5

    mps_dec = check_execution_admission(
        prob_large, "mps", "mps", mps_bond_dim=16
    )
    assert mps_dec.admitted is True
    assert mps_dec.num_qubits_allocated == 24

    # Reject when memory budget is tiny
    with pytest.raises(ResourceAdmissionError):
        check_execution_admission(
            prob_small,
            "bf_dcqo",
            "qiskit",
            budget=ResourceBudget(max_memory_mb=0.0001),
        )


def test_step7_recomputed_ablation_suite_and_overloaded_rejection() -> None:
    """Verify Step 7 benchmark records include full metadata, F(x) vs E(x), and overload rejection."""
    payload = run_full_ablation_suite(output_json=None, seeds=(42, 108))
    assert payload["infeasible_rejection_verified"] is True

    stats = {
        s["solver_label"]: s for s in payload["aggregated_statistics"]
    }
    assert "StructuredExact (Reference)" in stats
    assert "Random + MILP Repair" in stats
    assert "Joint Greedy + MILP Repair" in stats
    assert "BF-DCQO (Shot Sim) + Repair" in stats

    # Reference optimal operational cost F* must be <= every repaired heuristic mean F(x)
    f_opt = stats["StructuredExact (Reference)"]["mean_operational_cost_f"]
    f_rand = stats["Random + MILP Repair"]["mean_operational_cost_f"]
    f_bf = stats["BF-DCQO (Shot Sim) + Repair"]["mean_operational_cost_f"]
    assert f_opt is not None and f_rand is not None and f_bf is not None
    assert f_opt <= f_bf + 1e-6
    assert f_opt <= f_rand + 1e-6


def test_step9_shadow_mode_controller_evaluation() -> None:
    """Verify Step 9 shadow-mode evaluation across normal and overloaded slices."""
    metrics = run_shadow_mode_evaluation(num_slices=4, base_seed=500)
    by_ctrl = {m.controller_name: m for m in metrics}

    exact_m = by_ctrl["ExactReferenceController"]
    quantum_m = by_ctrl["QuantumBFDCQOWithRepairController"]

    assert exact_m.sla_throughput_pct == 100.0
    assert exact_m.overload_rejection_accuracy_pct == 100.0
    assert quantum_m.sla_throughput_pct == 100.0
    assert quantum_m.overload_rejection_accuracy_pct == 100.0
