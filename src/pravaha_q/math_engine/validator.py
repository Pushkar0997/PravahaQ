"""
Independent Physical & Constraint Validator
===========================================
Verifies candidate schedules solely against the CoupledInfrastructureProblem
specification (zero dependence on HUBO penalty weights or repair heuristics).

Checks:
  1. Bitstring dimension and binary values {0, 1}.
  2. Single-assignment per task: sum_n x_{m,n} == 1 for all m.
  3. Valid operating mode per node: sum_r y_{n,r} == 1 for all n (when R > 1).
  4. Linear mode-dependent capacity envelope:
         sum_m q_m x_{m,n} <= sum_r C_{n,r} y_{n,r}   for all n.
  5. Deadline feasibility under execution latency model:
         latency(m, n) <= deadline_m   for all assigned (m, n).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from pravaha_q.domain.problem_generator import CoupledInfrastructureProblem


class ScheduleValidity(str, Enum):
    """Whether a specific proposed schedule satisfies all physical/hard constraints."""

    VALID = "VALID"
    INVALID = "INVALID"


class SolverStatus(str, Enum):
    """What the solver/repair pipeline has established about the problem instance."""

    OPTIMAL = "OPTIMAL"
    FEASIBLE_INCUMBENT = "FEASIBLE_INCUMBENT"
    PROVEN_INFEASIBLE = "PROVEN_INFEASIBLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ConstraintViolation:
    """Detailed diagnostic record for a single violated hard constraint."""

    constraint_type: str
    entity_id: str
    observed_value: float
    limit_value: float
    margin: float  # Negative margin indicates violation magnitude
    description: str


@dataclass(frozen=True)
class ScheduleValidationReport:
    """Complete independent validation report for a proposed binary schedule."""

    validity: ScheduleValidity
    is_valid: bool
    operational_cost_f: float | None
    violations: tuple[ConstraintViolation, ...] = field(default_factory=tuple)
    node_loads_watts: dict[int, float] = field(default_factory=dict)
    node_capacities_watts: dict[int, float] = field(default_factory=dict)
    node_capacity_margins_watts: dict[int, float] = field(default_factory=dict)
    task_latency_margins_ms: dict[int, float] = field(default_factory=dict)
    task_to_node: dict[int, int] = field(default_factory=dict)
    node_to_mode: dict[int, int] = field(default_factory=dict)


class ScheduleValidator:
    """Optimizer-independent physical validator for CoupledInfrastructureProblem."""

    def __init__(self, problem: CoupledInfrastructureProblem, tol: float = 1e-6) -> None:
        self.problem = problem
        self.tol = tol

    def validate(self, candidate: Iterable[int] | str) -> ScheduleValidationReport:
        """Validate a binary vector or bitstring against all hard problem constraints."""
        if isinstance(candidate, str):
            try:
                x = [int(ch) for ch in candidate.strip()]
            except ValueError:
                return self._invalid_dimension_report("Non-integer characters in bitstring.")
        else:
            x = [int(v) for v in candidate]

        expected_len = self.problem.total_binary_variables
        if len(x) != expected_len:
            return self._invalid_dimension_report(
                f"Expected bitstring length {expected_len}, got {len(x)}."
            )

        if any(b not in (0, 1) for b in x):
            return self._invalid_dimension_report("Candidate contains non-binary values.")

        violations: list[ConstraintViolation] = []
        N_node = self.problem.num_nodes
        R = self.problem.num_power_states

        task_to_node: dict[int, int] = {}
        node_to_mode: dict[int, int] = {}
        node_loads: dict[int, float] = {n: 0.0 for n in range(N_node)}
        task_lat_margins: dict[int, float] = {}

        # 1. Task Single-Assignment & Deadline Check
        for m, task in enumerate(self.problem.workload.tasks):
            assigned_nodes = [
                n for n in range(N_node) if x[self.problem.task_node_var_index(m, n)] == 1
            ]
            count = len(assigned_nodes)
            if count != 1:
                violations.append(
                    ConstraintViolation(
                        constraint_type="TASK_ASSIGNMENT_ONE_HOT",
                        entity_id=f"task_{m}",
                        observed_value=float(count),
                        limit_value=1.0,
                        margin=-float(abs(count - 1)),
                        description=f"Task {m} assigned to {count} nodes (must be exactly 1).",
                    )
                )
            for n in assigned_nodes:
                task_to_node[m] = n
                node_loads[n] += task.thermal_load_watts
                exec_lat = self.problem.task_execution_latency_ms(m, n)
                lat_margin = round(task.deadline_ms - exec_lat, 4)
                task_lat_margins[m] = lat_margin
                if lat_margin < -self.tol:
                    violations.append(
                        ConstraintViolation(
                            constraint_type="TASK_DEADLINE_MS",
                            entity_id=f"task_{m}_on_node_{n}",
                            observed_value=exec_lat,
                            limit_value=task.deadline_ms,
                            margin=lat_margin,
                            description=(
                                f"Task {m} on Node {n} latency {exec_lat:.2f}ms exceeds "
                                f"deadline {task.deadline_ms:.2f}ms."
                            ),
                        )
                    )

        # 2. Node Operating Mode Selection Check
        node_caps: dict[int, float] = {}
        if self.problem.has_mode_variables:
            for n in range(N_node):
                active_modes = [
                    r for r in range(R) if x[self.problem.node_power_var_index(n, r)] == 1
                ]
                m_count = len(active_modes)
                if m_count != 1:
                    violations.append(
                        ConstraintViolation(
                            constraint_type="NODE_MODE_ONE_HOT",
                            entity_id=f"node_{n}",
                            observed_value=float(m_count),
                            limit_value=1.0,
                            margin=-float(abs(m_count - 1)),
                            description=f"Node {n} has {m_count} active modes (must be 1).",
                        )
                    )
                cap_n = sum(self.problem.node_mode_capacity(n, r) for r in active_modes)
                node_caps[n] = round(cap_n, 4)
                if m_count == 1:
                    node_to_mode[n] = active_modes[0]
        else:
            for n in range(N_node):
                node_to_mode[n] = 0
                node_caps[n] = self.problem.node_mode_capacity(n, 0)

        # 3. Linear Mode-Dependent Power/Capacity Envelope Check:
        #    sum_m q_m x_{m,n} <= sum_r C_{n,r} y_{n,r}
        cap_margins: dict[int, float] = {}
        for n in range(N_node):
            load_w = round(node_loads[n], 4)
            cap_w = node_caps[n]
            margin_w = round(cap_w - load_w, 4)
            cap_margins[n] = margin_w
            if margin_w < -self.tol:
                violations.append(
                    ConstraintViolation(
                        constraint_type="NODE_CAPACITY_ENVELOPE_WATTS",
                        entity_id=f"node_{n}",
                        observed_value=load_w,
                        limit_value=cap_w,
                        margin=margin_w,
                        description=(
                            f"Node {n} demand {load_w:.2f}W exceeds mode capacity "
                            f"envelope {cap_w:.2f}W (deficit: {margin_w:.2f}W)."
                        ),
                    )
                )

        is_valid = len(violations) == 0
        op_cost = self.problem.evaluate_operational_cost(x) if is_valid else None

        return ScheduleValidationReport(
            validity=ScheduleValidity.VALID if is_valid else ScheduleValidity.INVALID,
            is_valid=is_valid,
            operational_cost_f=op_cost,
            violations=tuple(violations),
            node_loads_watts={n: round(v, 2) for n, v in node_loads.items()},
            node_capacities_watts={n: round(v, 2) for n, v in node_caps.items()},
            node_capacity_margins_watts={n: round(v, 2) for n, v in cap_margins.items()},
            task_latency_margins_ms=task_lat_margins,
            task_to_node=task_to_node,
            node_to_mode=node_to_mode,
        )

    def _invalid_dimension_report(self, message: str) -> ScheduleValidationReport:
        return ScheduleValidationReport(
            validity=ScheduleValidity.INVALID,
            is_valid=False,
            operational_cost_f=None,
            violations=(
                ConstraintViolation(
                    constraint_type="BITSTRING_DOMAIN_ERROR",
                    entity_id="bitstring",
                    observed_value=0.0,
                    limit_value=float(self.problem.total_binary_variables),
                    margin=-1.0,
                    description=message,
                ),
            ),
        )
