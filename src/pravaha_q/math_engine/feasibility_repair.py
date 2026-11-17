"""
Constrained Minimum-Hamming-Distance Feasibility Repair Engine
==============================================================
Given a candidate bitstring x_hat in {0, 1}^N:
  1. Validates x_hat via the independent `ScheduleValidator`. If already valid,
     returns x_hat with `hamming_distance=0`.
  2. Otherwise, solves a constrained Mixed-Integer Linear Program (`scipy.optimize.milp`)
     subject to all hard constraints:
       - sum_n x_{m,n} == 1                           (for all tasks m)
       - sum_r y_{n,r} == 1                           (for all nodes n, if R > 1)
       - sum_m q_m x_{m,n} - sum_r C_{n,r} y_{n,r} <= 0  (mode capacity envelope)
       - x_{m,n} == 0 if latency(m, n) > deadline_m   (deadline compatibility)
     minimizing the linear Hamming distance from the proposed assignment:
       min  sum_{i=1}^N |x_i - x_hat_i| + eps * F_linear(x)
     where |x_i - x_hat_i| = (1 - 2*x_hat_i)*x_i + x_hat_i.
  3. If no feasible schedule exists within the time budget, returns explicit failure
     (`is_feasible=False`, `solver_status=PROVEN_INFEASIBLE` or `UNKNOWN`) and
     NEVER manufactures a success flag.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from pravaha_q.domain.problem_generator import CoupledInfrastructureProblem
from pravaha_q.math_engine.hubo_builder import HUBOPolynomial
from pravaha_q.math_engine.validator import (
    ScheduleValidationReport,
    ScheduleValidator,
    ScheduleValidity,
    SolverStatus,
)


@dataclass(frozen=True)
class DecodedInfrastructureSchedule:
    """Validated schedule and repair audit record."""

    raw_bitstring: str
    repaired_bitstring: str
    was_repaired: bool
    is_feasible: bool
    validity: ScheduleValidity
    solver_status: SolverStatus
    hamming_distance: int
    raw_operational_cost_f: float | None
    repaired_operational_cost_f: float | None
    raw_energy: float
    repaired_energy: float
    task_to_node: dict[int, int]
    node_to_power_state: dict[int, int]
    node_thermal_loads_watts: dict[int, float]
    node_thermal_ceilings_watts: dict[int, float]
    constraint_violations_fixed: int
    validation_report: ScheduleValidationReport | None = None
    repair_time_ms: float = 0.0


class FeasibilityRepairEngine:
    """Constrained MILP Minimum-Hamming-Distance Repair Engine."""

    def __init__(
        self,
        problem: CoupledInfrastructureProblem,
        hubo: HUBOPolynomial | None = None,
        time_limit_seconds: float = 0.25,
    ) -> None:
        self.problem = problem
        self.hubo = hubo
        self.time_limit_seconds = time_limit_seconds
        self.validator = ScheduleValidator(problem)

    def is_bitstring_feasible(self, bits: Iterable[int] | str) -> bool:
        """Delegate feasibility check strictly to the independent ScheduleValidator."""
        return self.validator.validate(bits).is_valid

    def repair_and_decode(
        self, raw_bits: Iterable[int] | str
    ) -> DecodedInfrastructureSchedule:
        """
        Validate `raw_bits` and, if invalid, solve the minimum-Hamming-distance
        constrained MILP repair problem. Never manufactures a fake success flag.
        """
        t0 = time.perf_counter()
        if isinstance(raw_bits, str):
            x_hat = [int(c) for c in raw_bits.strip()]
        else:
            x_hat = [int(b) for b in raw_bits]

        raw_str = "".join(str(b) for b in x_hat)
        raw_val_report = self.validator.validate(x_hat)
        raw_e = self.hubo.evaluate(x_hat) if self.hubo is not None else 0.0

        # Case 1: Candidate is already physically valid!
        if raw_val_report.is_valid:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return DecodedInfrastructureSchedule(
                raw_bitstring=raw_str,
                repaired_bitstring=raw_str,
                was_repaired=False,
                is_feasible=True,
                validity=ScheduleValidity.VALID,
                solver_status=SolverStatus.FEASIBLE_INCUMBENT,
                hamming_distance=0,
                raw_operational_cost_f=raw_val_report.operational_cost_f,
                repaired_operational_cost_f=raw_val_report.operational_cost_f,
                raw_energy=round(raw_e, 6),
                repaired_energy=round(raw_e, 6),
                task_to_node=raw_val_report.task_to_node,
                node_to_power_state=raw_val_report.node_to_mode,
                node_thermal_loads_watts=raw_val_report.node_loads_watts,
                node_thermal_ceilings_watts=raw_val_report.node_capacities_watts,
                constraint_violations_fixed=0,
                validation_report=raw_val_report,
                repair_time_ms=elapsed_ms,
            )

        # Case 2: Candidate is invalid -> Solve Constrained MILP Minimum-Hamming Repair
        repaired_bits, milp_status = self._solve_milp_minimum_hamming_repair(x_hat)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        if repaired_bits is None:
            return DecodedInfrastructureSchedule(
                raw_bitstring=raw_str,
                repaired_bitstring=raw_str,
                was_repaired=False,
                is_feasible=False,
                validity=ScheduleValidity.INVALID,
                solver_status=milp_status,
                hamming_distance=0,
                raw_operational_cost_f=None,
                repaired_operational_cost_f=None,
                raw_energy=round(raw_e, 6),
                repaired_energy=round(raw_e, 6),
                task_to_node=raw_val_report.task_to_node,
                node_to_power_state=raw_val_report.node_to_mode,
                node_thermal_loads_watts=raw_val_report.node_loads_watts,
                node_thermal_ceilings_watts=raw_val_report.node_capacities_watts,
                constraint_violations_fixed=0,
                validation_report=raw_val_report,
                repair_time_ms=elapsed_ms,
            )

        rep_report = self.validator.validate(repaired_bits)
        rep_str = "".join(str(b) for b in repaired_bits)
        rep_e = (
            self.hubo.evaluate(repaired_bits)
            if self.hubo is not None
            else (rep_report.operational_cost_f or 0.0)
        )
        hamming = int(sum(abs(a - b) for a, b in zip(repaired_bits, x_hat)))
        fixed_count = (
            len(raw_val_report.violations) - len(rep_report.violations)
            if rep_report.is_valid
            else 0
        )

        return DecodedInfrastructureSchedule(
            raw_bitstring=raw_str,
            repaired_bitstring=rep_str,
            was_repaired=(rep_str != raw_str and rep_report.is_valid),
            is_feasible=rep_report.is_valid,
            validity=rep_report.validity,
            solver_status=(
                SolverStatus.FEASIBLE_INCUMBENT
                if rep_report.is_valid
                else SolverStatus.UNKNOWN
            ),
            hamming_distance=hamming,
            raw_operational_cost_f=raw_val_report.operational_cost_f,
            repaired_operational_cost_f=rep_report.operational_cost_f,
            raw_energy=round(raw_e, 6),
            repaired_energy=round(rep_e, 6),
            task_to_node=rep_report.task_to_node,
            node_to_power_state=rep_report.node_to_mode,
            node_thermal_loads_watts=rep_report.node_loads_watts,
            node_thermal_ceilings_watts=rep_report.node_capacities_watts,
            constraint_violations_fixed=max(fixed_count, 0),
            validation_report=rep_report,
            repair_time_ms=elapsed_ms,
        )

    def _solve_milp_minimum_hamming_repair(
        self, x_hat: list[int]
    ) -> tuple[list[int] | None, SolverStatus]:
        """
        Solve:
            min  sum_i (1 - 2 * x_hat_i) * x_i + 1e-3 * c_op^T x
            s.t. A_eq x == b_eq,  A_ub x <= b_ub,  x in {0, 1}^N
        """
        N = self.problem.total_binary_variables
        M = self.problem.num_tasks
        N_node = self.problem.num_nodes
        R = self.problem.num_power_states

        c_obj = np.array([1.0 - 2.0 * float(b) for b in x_hat], dtype=np.float64)
        for m, task in enumerate(self.problem.workload.tasks):
            for n in range(N_node):
                idx = self.problem.task_node_var_index(m, n)
                c_obj[idx] += (
                    1e-3
                    * self.problem.task_execution_latency_ms(m, n)
                    * task.priority_weight
                )

        if self.problem.has_mode_variables:
            for n in range(N_node):
                for r, pstate in enumerate(self.problem.microgrid.power_states):
                    idx = self.problem.node_power_var_index(n, r)
                    c_obj[idx] += 1e-3 * pstate.energy_cost_tariff

        rows: list[np.ndarray] = []
        lbs: list[float] = []
        ubs: list[float] = []

        # 1. Task single-assignment equality: sum_n x_{m,n} == 1
        for m in range(M):
            row = np.zeros(N, dtype=np.float64)
            for n in range(N_node):
                row[self.problem.task_node_var_index(m, n)] = 1.0
            rows.append(row)
            lbs.append(1.0)
            ubs.append(1.0)

        # 2. Node mode selection equality: sum_r y_{n,r} == 1 (when R > 1)
        if self.problem.has_mode_variables:
            for n in range(N_node):
                row = np.zeros(N, dtype=np.float64)
                for r in range(R):
                    row[self.problem.node_power_var_index(n, r)] = 1.0
                rows.append(row)
                lbs.append(1.0)
                ubs.append(1.0)

        # 3. Mode-dependent capacity inequality: sum_m q_m x_{m,n} - sum_r C_{n,r} y_{n,r} <= 0
        for n in range(N_node):
            row = np.zeros(N, dtype=np.float64)
            for m, task in enumerate(self.problem.workload.tasks):
                row[self.problem.task_node_var_index(m, n)] = task.thermal_load_watts
            if self.problem.has_mode_variables:
                for r in range(R):
                    row[self.problem.node_power_var_index(n, r)] = (
                        -self.problem.node_mode_capacity(n, r)
                    )
                rows.append(row)
                lbs.append(-np.inf)
                ubs.append(0.0)
            else:
                rows.append(row)
                lbs.append(-np.inf)
                ubs.append(self.problem.node_mode_capacity(n, 0))

        # 4. Variable bounds (forbid task-node pairs that violate latency deadlines)
        var_ub = np.ones(N, dtype=np.float64)
        for m, task in enumerate(self.problem.workload.tasks):
            for n in range(N_node):
                if self.problem.task_execution_latency_ms(m, n) > task.deadline_ms + 1e-6:
                    var_ub[self.problem.task_node_var_index(m, n)] = 0.0

        A_mat = np.vstack(rows)
        constraints = LinearConstraint(A_mat, lbs, ubs)
        integrality = np.ones(N, dtype=np.float64)
        bounds = Bounds(np.zeros(N, dtype=np.float64), var_ub)

        res = milp(
            c=c_obj,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"time_limit": self.time_limit_seconds},
        )

        if res.success and res.x is not None:
            bits = [int(round(float(v))) for v in res.x]
            return bits, SolverStatus.FEASIBLE_INCUMBENT
        if res.status == 2:
            return None, SolverStatus.PROVEN_INFEASIBLE
        return None, SolverStatus.UNKNOWN
