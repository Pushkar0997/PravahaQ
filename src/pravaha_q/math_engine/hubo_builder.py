"""
Vaisheshika 3-Local HUBO (Higher-Order Unconstrained Binary Optimization) Compiler
==================================================================================
Constructs the coupled 1-body, 2-body quadratic, and 3-body cubic energy polynomial:
    E_HUBO(x) = c_0 + sum_i L_i x_i + sum_{i<j} Q_{ij} x_i x_j + sum_{i<j<k} T_{ijk} x_i x_j x_k
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
    Enforces RFC-0001 strict ascending index invariants: i < j and i < j < k.
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
            # Since x_i in {0, 1}, x_i^2 == x_i
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
        """Evaluate exact HUBO energy E_HUBO(x) for a binary vector x in {0, 1}^N."""
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
        """Return the N x N upper-triangular 2-body QUBO projection matrix."""
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
    Compiles a CoupledInfrastructureProblem into a 3-Local HUBOPolynomial.
    Named after Maharishi Kanada's Vaisheshika Sutra theory of 2-body (Dvyanuka)
    and 3-body (Tryanuka) interactions.
    """

    def __init__(
        self,
        penalty_config: UnbalancedPenaltyConfig | None = None,
        enable_cubic_terms: bool = True,
    ) -> None:
        self.penalty_config = penalty_config or UnbalancedPenaltyConfig()
        self.enable_cubic_terms = enable_cubic_terms

    def compile(self, problem: CoupledInfrastructureProblem) -> HUBOPolynomial:
        """Build the complete HUBOPolynomial for a coupled infrastructure instance."""
        hubo = HUBOPolynomial(
            num_variables=problem.total_binary_variables,
            variable_labels=problem.variable_labels(),
        )

        M = problem.num_tasks
        N_node = problem.num_nodes
        R = problem.num_power_states

        # 1. Linear Objective: Base task-node execution latency & priority
        for m, task in enumerate(problem.workload.tasks):
            for n, node in enumerate(problem.cluster.nodes):
                idx = problem.task_node_var_index(m, n)
                cost = (
                    node.base_latency_ms
                    * (task.compute_flops_giga / 50.0)
                    * task.priority_weight
                )
                hubo.add_linear(idx, cost)

        # 2. Linear Objective: Microgrid power dispatch state tariff
        for n in range(N_node):
            for r, pstate in enumerate(problem.microgrid.power_states):
                idx = problem.node_power_var_index(n, r)
                hubo.add_linear(idx, pstate.energy_cost_tariff)

        # 3. Quadratic Objective: Inter-node KV-cache synchronization overhead
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                w_sync = problem.workload.sync_coupling(m1, m2)
                if w_sync > 0:
                    for n1 in range(N_node):
                        for n2 in range(N_node):
                            dist = problem.cluster.distance(n1, n2)
                            if dist > 0:
                                i = problem.task_node_var_index(m1, n1)
                                j = problem.task_node_var_index(m2, n2)
                                hubo.add_quadratic(i, j, w_sync * dist)

        # 4. Equality Constraints: Each task m assigned to exactly 1 node
        #    lambda_eq * (sum_n x_{m,n} - 1)^2 = lambda_eq * (1 - sum_n x_{m,n} + 2 sum_{n1<n2} x_{m,n1} x_{m,n2})
        lam_eq = self.penalty_config.equality_weight
        for m in range(M):
            hubo.add_constant(lam_eq)
            for n1 in range(N_node):
                i = problem.task_node_var_index(m, n1)
                hubo.add_linear(i, -lam_eq)
                for n2 in range(n1 + 1, N_node):
                    j = problem.task_node_var_index(m, n2)
                    hubo.add_quadratic(i, j, 2.0 * lam_eq)

        # 5. Equality Constraints: Each node n operates in exactly 1 power state r
        if R > 1:
            for n in range(N_node):
                hubo.add_constant(lam_eq)
                for r1 in range(R):
                    i = problem.node_power_var_index(n, r1)
                    hubo.add_linear(i, -lam_eq)
                    for r2 in range(r1 + 1, R):
                        j = problem.node_power_var_index(n, r2)
                        hubo.add_quadratic(i, j, 2.0 * lam_eq)

        # 6. Slack-Free Unbalanced Inequality Penalties for Node Thermal Capacity
        for n, node in enumerate(problem.cluster.nodes):
            var_indices = tuple(problem.task_node_var_index(m, n) for m in range(M))
            weights = tuple(t.thermal_load_watts for t in problem.workload.tasks)
            spec = InequalityConstraintSpec(
                name=f"ThermalCap_Node_{n}",
                var_indices=var_indices,
                weights=weights,
                capacity=node.thermal_capacity_watts,
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

        # 7. Vaisheshika 3-Body Cubic Terms (T_{ijk} * x_i * x_j * x_k):
        #    Penalize simultaneous co-execution of two heavy tasks on a node while
        #    that node's microgrid slice is in Eco-Throttled cooling state!
        if self.enable_cubic_terms and R >= 1:
            for n, node in enumerate(problem.cluster.nodes):
                for r, pstate in enumerate(problem.microgrid.power_states):
                    if pstate.is_eco_throttled:
                        idx_power = problem.node_power_var_index(n, r)
                        for m1 in range(M):
                            for m2 in range(m1 + 1, M):
                                combined_heat = (
                                    problem.workload.tasks[m1].thermal_load_watts
                                    + problem.workload.tasks[m2].thermal_load_watts
                                )
                                eco_ceiling = (
                                    node.thermal_capacity_watts * pstate.cooling_multiplier
                                )
                                if combined_heat > 0.85 * eco_ceiling:
                                    idx_t1 = problem.task_node_var_index(m1, n)
                                    idx_t2 = problem.task_node_var_index(m2, n)
                                    overload_ratio = combined_heat / max(eco_ceiling, 1.0)
                                    cubic_penalty = (
                                        problem.microgrid.phase_overload_penalty
                                        * overload_ratio
                                    )
                                    hubo.add_cubic(idx_t1, idx_t2, idx_power, cubic_penalty)

        return hubo
