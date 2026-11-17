"""
Vaisheshika QUBO / HUBO Compiler
================================
Compiles CoupledInfrastructureProblem into the penalized surrogate polynomial:
    E(x) = F(x) + Phi_penalties(x)
where:
  - F(x) is the true operational cost (latency + KV sync + mode tariff).
  - Mode-dependent capacity g_n(x, y) = sum_r C_{n,r} y_{n,r} - sum_m q_m x_{m,n} >= 0
    is linear in the binary variables and encoded via 2-body unbalanced penalization.
  - Cubic 3-body terms are disabled by default (`enable_cubic_terms=False`) and only
    included when explicitly studying higher-order co-residency interference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from pravaha_q.config import UnbalancedPenaltyConfig
from pravaha_q.domain.exceptions import HUBOCompilationError
from pravaha_q.domain.problem_generator import CoupledInfrastructureProblem
from pravaha_q.math_engine.unbalanced_penalties import (
    InequalityConstraintSpec,
    compile_unbalanced_inequality_penalty,
)


@dataclass
class HUBOPolynomial:
    """
    Canonical up-to-3-body pseudo-Boolean polynomial over x in {0, 1}^N.
    Enforces strict ascending index invariants: i < j and i < j < k.
    """

    num_variables: int
    constant_offset: float = 0.0
    linear: dict[int, float] = field(default_factory=dict)
    quadratic: dict[tuple[int, int], float] = field(default_factory=dict)
    cubic: dict[tuple[int, int, int], float] = field(default_factory=dict)
    variable_labels: list[str] = field(default_factory=list)

    def add_constant(self, val: float) -> None:
        self.constant_offset += float(val)

    def add_linear(self, i: int, val: float) -> None:
        self._check_idx(i)
        if abs(val) > 1e-15:
            self.linear[i] = self.linear.get(i, 0.0) + float(val)

    def add_quadratic(self, i: int, j: int, val: float) -> None:
        self._check_idx(i)
        self._check_idx(j)
        if i == j:
            self.add_linear(i, val)
            return
        pair = (min(i, j), max(i, j))
        if abs(val) > 1e-15:
            self.quadratic[pair] = self.quadratic.get(pair, 0.0) + float(val)

    def add_cubic(self, i: int, j: int, k: int, val: float) -> None:
        self._check_idx(i)
        self._check_idx(j)
        self._check_idx(k)
        unique = sorted({i, j, k})
        if len(unique) == 1:
            self.add_linear(unique[0], val)
        elif len(unique) == 2:
            self.add_quadratic(unique[0], unique[1], val)
        else:
            triad = (unique[0], unique[1], unique[2])
            if abs(val) > 1e-15:
                self.cubic[triad] = self.cubic.get(triad, 0.0) + float(val)

    def evaluate(self, bitstring: Iterable[int] | np.ndarray | str) -> float:
        """Evaluate penalized surrogate energy E(x) = F(x) + penalties."""
        x = self._parse_bitstring(bitstring)
        energy = float(self.constant_offset)
        for i, coeff in self.linear.items():
            if x[i]:
                energy += coeff
        for (i, j), coeff in self.quadratic.items():
            if x[i] and x[j]:
                energy += coeff
        for (i, j, k), coeff in self.cubic.items():
            if x[i] and x[j] and x[k]:
                energy += coeff
        return energy

    def to_qubo_matrix(self) -> np.ndarray:
        q_mat = np.zeros((self.num_variables, self.num_variables), dtype=np.float64)
        for i, val in self.linear.items():
            q_mat[i, i] = val
        for (i, j), val in self.quadratic.items():
            q_mat[i, j] = val
        return q_mat

    def _check_idx(self, idx: int) -> None:
        if not (0 <= idx < self.num_variables):
            raise HUBOCompilationError(
                f"Variable index {idx} out of bounds for N={self.num_variables}."
            )

    def _parse_bitstring(self, bitstring: Iterable[int] | np.ndarray | str) -> list[int]:
        if isinstance(bitstring, str):
            bits = [int(ch) for ch in bitstring.strip()]
        else:
            bits = [int(v) for v in bitstring]
        if len(bits) != self.num_variables:
            raise HUBOCompilationError(
                f"Expected bitstring of length {self.num_variables}, got {len(bits)}."
            )
        return bits


class VaisheshikaHUBOCompiler:
    """
    Compiles a CoupledInfrastructureProblem into a QUBO (default) or 3-Local HUBO
    (when `enable_cubic_terms=True` is explicitly requested).
    """

    def __init__(
        self,
        penalty_config: UnbalancedPenaltyConfig | None = None,
        enable_cubic_terms: bool = False,
    ) -> None:
        self.penalty_config = penalty_config or UnbalancedPenaltyConfig()
        self.enable_cubic_terms = enable_cubic_terms

    def compile(self, problem: CoupledInfrastructureProblem) -> HUBOPolynomial:
        hubo = HUBOPolynomial(
            num_variables=problem.total_binary_variables,
            variable_labels=problem.variable_labels(),
        )

        M = problem.num_tasks
        N_node = problem.num_nodes
        R = problem.num_power_states
        lam_eq = self.penalty_config.equality_weight

        # 1. Operational Objective F(x): Task execution latency + deadline violation penalty
        for m, task in enumerate(problem.workload.tasks):
            for n in range(N_node):
                idx = problem.task_node_var_index(m, n)
                exec_lat = problem.task_execution_latency_ms(m, n)
                cost = exec_lat * task.priority_weight
                if exec_lat > task.deadline_ms:
                    cost += 2.0 * lam_eq
                hubo.add_linear(idx, cost)

        # 2. Operational Objective F(x): Mode tariff cost
        if problem.has_mode_variables:
            for n in range(N_node):
                for r, pstate in enumerate(problem.microgrid.power_states):
                    idx = problem.node_power_var_index(n, r)
                    hubo.add_linear(idx, pstate.energy_cost_tariff)
        else:
            # R == 1: y_{n,0} = 1 is fixed; add constant tariff offset
            hubo.add_constant(N_node * problem.microgrid.power_states[0].energy_cost_tariff)

        # 3. Operational Objective F(x): Inter-node KV-cache synchronization overhead
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                w_sync = problem.workload.sync_coupling(m1, m2)
                if w_sync > 0.0:
                    for n1 in range(N_node):
                        for n2 in range(N_node):
                            dist = problem.cluster.distance(n1, n2)
                            if dist > 0.0:
                                i = problem.task_node_var_index(m1, n1)
                                j = problem.task_node_var_index(m2, n2)
                                hubo.add_quadratic(i, j, w_sync * dist)

        # 4. Hard Equality Penalty: sum_n x_{m,n} == 1 for all m
        for m in range(M):
            hubo.add_constant(lam_eq)
            for n1 in range(N_node):
                i = problem.task_node_var_index(m, n1)
                hubo.add_linear(i, -lam_eq)
                for n2 in range(n1 + 1, N_node):
                    j = problem.task_node_var_index(m, n2)
                    hubo.add_quadratic(i, j, 2.0 * lam_eq)

        # 5. Hard Equality Penalty: sum_r y_{n,r} == 1 for all n (only when R > 1)
        if problem.has_mode_variables:
            for n in range(N_node):
                hubo.add_constant(lam_eq)
                for r1 in range(R):
                    i = problem.node_power_var_index(n, r1)
                    hubo.add_linear(i, -lam_eq)
                    for r2 in range(r1 + 1, R):
                        j = problem.node_power_var_index(n, r2)
                        hubo.add_quadratic(i, j, 2.0 * lam_eq)

        # 6. Linear Mode-Dependent Capacity Inequality Penalty:
        #    sum_m q_m x_{m,n} - sum_r C_{n,r} y_{n,r} <= 0
        for n in range(N_node):
            if problem.has_mode_variables:
                # Write as: C_{n,max} - [ sum_m q_m x_{m,n} + sum_r (C_{n,max} - C_{n,r}) y_{n,r} ] >= 0
                c_max = max(problem.node_mode_capacity(n, r) for r in range(R))
                var_indices = [problem.task_node_var_index(m, n) for m in range(M)] + [
                    problem.node_power_var_index(n, r) for r in range(R)
                ]
                weights = [t.thermal_load_watts for t in problem.workload.tasks] + [
                    c_max - problem.node_mode_capacity(n, r) for r in range(R)
                ]
                spec = InequalityConstraintSpec(
                    name=f"ModeCapacity_Node_{n}",
                    var_indices=tuple(var_indices),
                    weights=tuple(weights),
                    capacity=c_max,
                )
            else:
                spec = InequalityConstraintSpec(
                    name=f"Capacity_Node_{n}",
                    var_indices=tuple(problem.task_node_var_index(m, n) for m in range(M)),
                    weights=tuple(t.thermal_load_watts for t in problem.workload.tasks),
                    capacity=problem.node_mode_capacity(n, 0),
                )

            c_off, lin_map, quad_map = compile_unbalanced_inequality_penalty(
                spec,
                lambda1=self.penalty_config.lambda1_linear,
                lambda2=self.penalty_config.lambda2_quadratic,
            )
            hubo.add_constant(c_off)
            for idx, val in lin_map.items():
                hubo.add_linear(idx, val)
            for (i, j), val in quad_map.items():
                hubo.add_quadratic(i, j, val)

        # 7. Optional Higher-Order Co-Residency Interference (Only when enable_cubic_terms=True)
        if self.enable_cubic_terms and problem.has_mode_variables:
            for n in range(N_node):
                for r, pstate in enumerate(problem.microgrid.power_states):
                    if pstate.is_eco_throttled:
                        idx_power = problem.node_power_var_index(n, r)
                        cap_nr = problem.node_mode_capacity(n, r)
                        for m1 in range(M):
                            for m2 in range(m1 + 1, M):
                                combined_load = (
                                    problem.workload.tasks[m1].thermal_load_watts
                                    + problem.workload.tasks[m2].thermal_load_watts
                                )
                                if combined_load > 0.85 * cap_nr:
                                    idx_t1 = problem.task_node_var_index(m1, n)
                                    idx_t2 = problem.task_node_var_index(m2, n)
                                    ratio = combined_load / max(cap_nr, 1.0)
                                    hubo.add_cubic(
                                        idx_t1,
                                        idx_t2,
                                        idx_power,
                                        problem.microgrid.phase_overload_penalty * ratio,
                                    )

        return hubo
