"""
Deterministic Post-Quantum Feasibility Repair Engine (RFC-0003)
===============================================================
Guarantees that 100% of schedules returned by PravahaQ obey:
  1. Single-Assignment Invariant: Every task m is routed to exactly 1 node n.
  2. Power-State One-Hot Invariant: Every node n operates in exactly 1 power state r.
  3. Thermal Headroom Preference: Upgrades power state or migrates tasks if a node
     exceeds its active thermal cooling envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pravaha_q.domain.problem_generator import CoupledInfrastructureProblem
from pravaha_q.math_engine.hubo_builder import HUBOPolynomial


@dataclass(frozen=True)
class DecodedInfrastructureSchedule:
    """Human-readable, verified schedule decoded from a quantum bitstring."""

    raw_bitstring: str
    repaired_bitstring: str
    was_repaired: bool
    is_feasible: bool
    raw_energy: float
    repaired_energy: float
    task_to_node: dict[int, int]
    node_to_power_state: dict[int, int]
    node_thermal_loads_watts: dict[int, float]
    node_thermal_ceilings_watts: dict[int, float]
    constraint_violations_fixed: int


class FeasibilityRepairEngine:
    """Implements the Two-Phase Deterministic Feasibility Repair algorithm (RFC-0003)."""

    def __init__(
        self,
        problem: CoupledInfrastructureProblem,
        hubo: HUBOPolynomial,
    ) -> None:
        self.problem = problem
        self.hubo = hubo

    def is_bitstring_feasible(self, bits: Iterable[int] | str) -> bool:
        """Check if a bitstring satisfies all one-hot assignment and power state rules."""
        x = [int(c) for c in bits] if isinstance(bits, str) else [int(b) for b in bits]
        M = self.problem.num_tasks
        N_node = self.problem.num_nodes
        R = self.problem.num_power_states

        for m in range(M):
            assigned = sum(
                x[self.problem.task_node_var_index(m, n)] for n in range(N_node)
            )
            if assigned != 1:
                return False

        for n in range(N_node):
            p_assigned = sum(
                x[self.problem.node_power_var_index(n, r)] for r in range(R)
            )
            if p_assigned != 1:
                return False

        return True

    def repair_and_decode(
        self, raw_bits: Iterable[int] | str
    ) -> DecodedInfrastructureSchedule:
        """
        Repair any constraint violations in O(M * N_node) time and return a
        guaranteed-valid DecodedInfrastructureSchedule.
        """
        if isinstance(raw_bits, str):
            x_raw = [int(c) for c in raw_bits.strip()]
        else:
            x_raw = [int(b) for b in raw_bits]

        raw_str = "".join(str(b) for b in x_raw)
        raw_energy = self.hubo.evaluate(x_raw)

        M = self.problem.num_tasks
        N_node = self.problem.num_nodes
        R = self.problem.num_power_states

        violations_fixed = 0
        task_to_node: dict[int, int] = {}
        node_loads: dict[int, float] = {n: 0.0 for n in range(N_node)}

        # Phase 1A: Project each task m onto exactly one node n
        for m, task in enumerate(self.problem.workload.tasks):
            active_nodes = [
                n
                for n in range(N_node)
                if x_raw[self.problem.task_node_var_index(m, n)] == 1
            ]
            if len(active_nodes) == 1:
                chosen_n = active_nodes[0]
            else:
                violations_fixed += 1
                candidates = active_nodes if len(active_nodes) > 1 else list(range(N_node))
                # Pick node minimizing base latency + thermal congestion ratio
                chosen_n = min(
                    candidates,
                    key=lambda n: (
                        self.problem.cluster.nodes[n].base_latency_ms
                        + 3.0
                        * (node_loads[n] + task.thermal_load_watts)
                        / self.problem.cluster.nodes[n].thermal_capacity_watts
                    ),
                )
            task_to_node[m] = chosen_n
            node_loads[chosen_n] += task.thermal_load_watts

        # Phase 1B: Project each node n onto a valid power state r
        node_to_power: dict[int, int] = {}
        for n, node in enumerate(self.problem.cluster.nodes):
            active_r = [
                r
                for r in range(R)
                if x_raw[self.problem.node_power_var_index(n, r)] == 1
            ]
            if len(active_r) == 1:
                chosen_r = active_r[0]
            else:
                violations_fixed += 1
                chosen_r = 0

            # Phase 2: Thermal headroom check — if eco state ceiling is breached, upgrade state
            eco_ceiling = (
                node.thermal_capacity_watts
                * self.problem.microgrid.power_states[chosen_r].cooling_multiplier
            )
            if node_loads[n] > eco_ceiling and R > 1:
                best_r = max(
                    range(R),
                    key=lambda r: self.problem.microgrid.power_states[r].cooling_multiplier,
                )
                if best_r != chosen_r:
                    chosen_r = best_r
                    violations_fixed += 1

            node_to_power[n] = chosen_r

        # Phase 2B: Rebalance tasks if a node still breaches its maximum thermal ceiling
        for n, node in enumerate(self.problem.cluster.nodes):
            ceil_n = (
                node.thermal_capacity_watts
                * self.problem.microgrid.power_states[node_to_power[n]].cooling_multiplier
            )
            if node_loads[n] > ceil_n and N_node > 1:
                tasks_on_n = [m for m, assigned_n in task_to_node.items() if assigned_n == n]
                tasks_on_n.sort(
                    key=lambda m: self.problem.workload.tasks[m].thermal_load_watts,
                    reverse=True,
                )
                for m_move in tasks_on_n:
                    if node_loads[n] <= ceil_n:
                        break
                    t_heat = self.problem.workload.tasks[m_move].thermal_load_watts
                    # Find another node with sufficient thermal slack
                    other_nodes = [
                        n_alt
                        for n_alt in range(N_node)
                        if n_alt != n
                        and (node_loads[n_alt] + t_heat)
                        <= (
                            self.problem.cluster.nodes[n_alt].thermal_capacity_watts
                            * self.problem.microgrid.power_states[
                                node_to_power[n_alt]
                            ].cooling_multiplier
                        )
                    ]
                    if other_nodes:
                        target_n = min(other_nodes, key=lambda na: node_loads[na])
                        task_to_node[m_move] = target_n
                        node_loads[n] -= t_heat
                        node_loads[target_n] += t_heat
                        violations_fixed += 1

        # Reconstruct canonical repaired binary vector x_rep
        x_rep = [0] * self.problem.total_binary_variables
        for m, n in task_to_node.items():
            x_rep[self.problem.task_node_var_index(m, n)] = 1
        for n, r in node_to_power.items():
            x_rep[self.problem.node_power_var_index(n, r)] = 1

        rep_str = "".join(str(b) for b in x_rep)
        rep_energy = self.hubo.evaluate(x_rep)
        ceilings = {
            n: round(
                self.problem.cluster.nodes[n].thermal_capacity_watts
                * self.problem.microgrid.power_states[node_to_power[n]].cooling_multiplier,
                2,
            )
            for n in range(N_node)
        }
        loads_rounded = {n: round(val, 2) for n, val in node_loads.items()}

        return DecodedInfrastructureSchedule(
            raw_bitstring=raw_str,
            repaired_bitstring=rep_str,
            was_repaired=(raw_str != rep_str),
            is_feasible=self.is_bitstring_feasible(x_rep),
            raw_energy=round(raw_energy, 6),
            repaired_energy=round(rep_energy, 6),
            task_to_node=task_to_node,
            node_to_power_state=node_to_power,
            node_thermal_loads_watts=loads_rounded,
            node_thermal_ceilings_watts=ceilings,
            constraint_violations_fixed=violations_fixed,
        )
