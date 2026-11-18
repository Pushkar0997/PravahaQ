"""
Controlled Shadow-Mode Cluster Evaluator (Step 9 of GPT-6 Astra Audit)
======================================================================
Replays a recorded sequence of workload & microgrid telemetry slices (including
normal demand and sudden overload spikes) through competing schedulers in
shadow mode:
  1. `SeparateLegacyController`: Decoupled routing + mode selection (no joint repair).
  2. `GreedyWithRepairController`: Joint greedy heuristic + MILP repair.
  3. `QuantumBFDCQOWithRepairController`: Shot-simulated BF-DCQO + MILP repair.
  4. `ExactReferenceController`: Structured enumeration ground truth.

Measures:
  - Mean operational cost F(x) per successfully scheduled slice
  - SLA Throughput (% of feasible slices scheduled within latency/capacity limits)
  - Overload Rejection Accuracy (% of truly infeasible slices honestly rejected)
  - p95 and p99 scheduling overhead latency (ms)
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from pravaha_q.algorithms.bf_dcqo import BFDCQOEngine
from pravaha_q.benchmarks.suite_runner import solve_separate_routing_then_mode
from pravaha_q.config import BFDCQOConfig
from pravaha_q.domain.problem_generator import (
    CoupledInfrastructureProblem,
    generate_coupled_problem,
)
from pravaha_q.math_engine.feasibility_repair import FeasibilityRepairEngine
from pravaha_q.math_engine.hubo_builder import VaisheshikaHUBOCompiler
from pravaha_q.math_engine.ising_spin_glass import hubo_to_ising
from pravaha_q.math_engine.validator import ScheduleValidator, SolverStatus
from pravaha_q.solvers.classical_baselines import (
    GreedyOperationalSolver,
    StructuredEnumerationSolver,
)


@dataclass(frozen=True)
class ShadowControllerMetrics:
    """Operational telemetry collected for a single controller in shadow mode."""

    controller_name: str
    total_slices: int
    feasible_slices_count: int
    infeasible_slices_count: int
    valid_schedules_returned: int
    sla_throughput_pct: float
    overload_rejection_accuracy_pct: float
    mean_operational_cost_f: float | None
    p95_scheduling_latency_ms: float
    p99_scheduling_latency_ms: float


def generate_recorded_workload_trace(
    num_slices: int = 6,
    base_seed: int = 500,
) -> list[tuple[CoupledInfrastructureProblem, bool]]:
    """
    Generate a multi-slice recorded cluster trace where slices 0..num_slices-2 are
    normal/high load (`capacity_scale=1.0`) and the final slice is a deliberate
    device-overload spike (`capacity_scale=0.15`, truly infeasible).
    Returns list of `(problem_slice, is_truly_feasible)`.
    """
    trace: list[tuple[CoupledInfrastructureProblem, bool]] = []
    for idx in range(num_slices):
        is_overload = idx == (num_slices - 1)
        scale = 0.15 if is_overload else 1.0
        prob = generate_coupled_problem(
            num_nodes=2,
            num_tasks=3,
            num_power_states=2,
            seed=base_seed + idx * 17,
            name=f"ShadowSlice-{idx}-{'OVERLOAD' if is_overload else 'NORMAL'}",
            capacity_scale=scale,
        )
        enum_check = StructuredEnumerationSolver().solve_problem(prob)
        truly_feasible = enum_check.solver_status == SolverStatus.OPTIMAL
        trace.append((prob, truly_feasible))
    return trace


def run_shadow_mode_evaluation(
    num_slices: int = 5,
    base_seed: int = 500,
) -> list[ShadowControllerMetrics]:
    """Replay the recorded workload trace across all 4 competing controllers."""
    trace = generate_recorded_workload_trace(
        num_slices=num_slices, base_seed=base_seed
    )
    controllers = (
        "SeparateLegacyController",
        "GreedyWithRepairController",
        "QuantumBFDCQOWithRepairController",
        "ExactReferenceController",
    )

    results: list[ShadowControllerMetrics] = []
    total_feasible = sum(1 for _, feas in trace if feas)
    total_infeasible = sum(1 for _, feas in trace if not feas)

    for ctrl_name in controllers:
        valid_count = 0
        correct_rejections = 0
        op_costs: list[float] = []
        latencies_ms: list[float] = []

        for idx, (prob, truly_feasible) in enumerate(trace):
            t0 = time.perf_counter()
            hubo = VaisheshikaHUBOCompiler().compile(prob)
            validator = ScheduleValidator(prob)
            repair_engine = FeasibilityRepairEngine(prob, hubo)

            if ctrl_name == "SeparateLegacyController":
                bits = solve_separate_routing_then_mode(prob)
                val_rep = validator.validate(bits)
                is_valid = val_rep.is_valid
                cost_f = val_rep.operational_cost_f
                rejected = not is_valid
            elif ctrl_name == "GreedyWithRepairController":
                g_res = GreedyOperationalSolver().solve_problem(prob, hubo)
                dec = repair_engine.repair_and_decode(g_res.best_bitstring)
                is_valid = dec.is_feasible
                cost_f = dec.repaired_operational_cost_f
                rejected = (
                    not dec.is_feasible
                    and dec.solver_status == SolverStatus.PROVEN_INFEASIBLE
                )
            elif ctrl_name == "QuantumBFDCQOWithRepairController":
                ising = hubo_to_ising(hubo)
                bf = BFDCQOEngine(
                    config=BFDCQOConfig(max_iterations=3),
                    shots=512,
                    seed=base_seed + idx,
                    use_shot_estimates=True,
                ).run(ising)
                dec = repair_engine.repair_and_decode(bf.best_bitstring)
                is_valid = dec.is_feasible
                cost_f = dec.repaired_operational_cost_f
                rejected = (
                    not dec.is_feasible
                    and dec.solver_status == SolverStatus.PROVEN_INFEASIBLE
                )
            else:
                ex_res = StructuredEnumerationSolver().solve_problem(prob, hubo)
                is_valid = ex_res.validity.value == "VALID"
                cost_f = ex_res.operational_cost_f
                rejected = ex_res.solver_status == SolverStatus.PROVEN_INFEASIBLE

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            if truly_feasible and is_valid and cost_f is not None:
                valid_count += 1
                op_costs.append(cost_f)
            elif (not truly_feasible) and rejected:
                correct_rejections += 1

        sla_pct = (
            100.0 * valid_count / max(total_feasible, 1)
            if total_feasible > 0
            else 100.0
        )
        rej_pct = (
            100.0 * correct_rejections / max(total_infeasible, 1)
            if total_infeasible > 0
            else 100.0
        )
        mean_f = round(float(np.mean(op_costs)), 4) if op_costs else None

        results.append(
            ShadowControllerMetrics(
                controller_name=ctrl_name,
                total_slices=len(trace),
                feasible_slices_count=total_feasible,
                infeasible_slices_count=total_infeasible,
                valid_schedules_returned=valid_count,
                sla_throughput_pct=round(sla_pct, 2),
                overload_rejection_accuracy_pct=round(rej_pct, 2),
                mean_operational_cost_f=mean_f,
                p95_scheduling_latency_ms=round(
                    float(np.percentile(latencies_ms, 95)), 2
                ),
                p99_scheduling_latency_ms=round(
                    float(np.percentile(latencies_ms, 99)), 2
                ),
            )
        )

    return results
