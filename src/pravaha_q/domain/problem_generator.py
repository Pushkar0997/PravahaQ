"""Deterministic generator and mathematical specification for Coupled Routing & Mode Selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pravaha_q.domain.cluster_topology import EdgeClusterTopology, EdgeComputeNode
from pravaha_q.domain.microgrid_physics import MicrogridPowerState, MicrogridSlice
from pravaha_q.domain.workload_trace import InferenceBatchTask, WorkloadGraph


@dataclass(frozen=True)
class CoupledInfrastructureProblem:
    """
    Mathematical Problem Specification (Independent of Solver Encoding):
      - M tasks (m = 0..M-1) with demand q_m (watts) and deadline D_m (ms).
      - N_node nodes (n = 0..N_node-1) and R operating modes (r = 0..R-1).
      - Capacity of node n in mode r: C_{n,r} = base_capacity_n * mode_multiplier_r.
      - Binary routing variables: x_{m,n} in {0, 1} (indices 0 .. M*N_node - 1).
      - Binary mode variables:
          * If R > 1: y_{n,r} in {0, 1} (indices M*N_node .. M*N_node + N_node*R - 1).
          * If R == 1: y_{n,0} = 1 is fixed and ELIMINATED from the binary decision vector.
    """

    name: str
    cluster: EdgeClusterTopology
    workload: WorkloadGraph
    microgrid: MicrogridSlice

    @property
    def num_tasks(self) -> int:
        return self.workload.num_tasks

    @property
    def num_nodes(self) -> int:
        return self.cluster.num_nodes

    @property
    def num_power_states(self) -> int:
        return self.microgrid.num_states

    @property
    def has_mode_variables(self) -> bool:
        """When R == 1, y_{n,0} = 1 is substituted out to eliminate unconstrained variables."""
        return self.num_power_states > 1

    @property
    def num_routing_vars(self) -> int:
        return self.num_tasks * self.num_nodes

    @property
    def num_power_vars(self) -> int:
        return self.num_nodes * self.num_power_states if self.has_mode_variables else 0

    @property
    def total_binary_variables(self) -> int:
        return self.num_routing_vars + self.num_power_vars

    def task_node_var_index(self, task_m: int, node_n: int) -> int:
        if not (0 <= task_m < self.num_tasks and 0 <= node_n < self.num_nodes):
            raise IndexError(f"Invalid (task_m={task_m}, node_n={node_n}).")
        return task_m * self.num_nodes + node_n

    def node_power_var_index(self, node_n: int, state_r: int) -> int:
        if not self.has_mode_variables:
            raise ValueError(
                "Mode variables y_{n,r} are eliminated when num_power_states == 1 (y_{n,0}=1 fixed)."
            )
        if not (0 <= node_n < self.num_nodes and 0 <= state_r < self.num_power_states):
            raise IndexError(f"Invalid (node_n={node_n}, state_r={state_r}).")
        return self.num_routing_vars + node_n * self.num_power_states + state_r

    def node_mode_capacity(self, node_n: int, state_r: int = 0) -> float:
        """Return capacity envelope C_{n,r} (watts) of node n in operating mode r."""
        base_cap = self.cluster.nodes[node_n].thermal_capacity_watts
        mult = self.microgrid.power_states[state_r].cooling_multiplier
        return round(float(base_cap * mult), 4)

    def task_execution_latency_ms(self, task_m: int, node_n: int) -> float:
        """Return execution latency (ms) of task m on node n."""
        task = self.workload.tasks[task_m]
        node = self.cluster.nodes[node_n]
        return round(float(node.base_latency_ms * (task.compute_flops_giga / 50.0)), 4)

    def evaluate_operational_cost(self, bitstring: list[int] | tuple[int, ...] | str) -> float:
        """
        Compute the true Operational Cost F(x) (independent of encoding penalties):
          F(x) = sum_{m,n} c_{m,n} x_{m,n}
               + sum_{m<m'} sum_{n,n'} W_{m,m'} D_{n,n'} x_{m,n} x_{m',n'}
               + sum_{n,r} P_{n,r} y_{n,r}
        """
        x = (
            [int(c) for c in bitstring.strip()]
            if isinstance(bitstring, str)
            else [int(b) for b in bitstring]
        )
        if len(x) != self.total_binary_variables:
            raise ValueError(f"Expected {self.total_binary_variables} bits, got {len(x)}.")

        cost = 0.0
        M, N_node, R = self.num_tasks, self.num_nodes, self.num_power_states

        # 1. Task execution latency cost weighted by priority
        for m, task in enumerate(self.workload.tasks):
            for n in range(N_node):
                if x[self.task_node_var_index(m, n)] == 1:
                    cost += self.task_execution_latency_ms(m, n) * task.priority_weight

        # 2. Inter-node KV-cache synchronization overhead
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                w_sync = self.workload.sync_coupling(m1, m2)
                if w_sync > 0.0:
                    for n1 in range(N_node):
                        if x[self.task_node_var_index(m1, n1)] == 0:
                            continue
                        for n2 in range(N_node):
                            if x[self.task_node_var_index(m2, n2)] == 1:
                                cost += w_sync * self.cluster.distance(n1, n2)

        # 3. Operating mode energy tariff cost
        if self.has_mode_variables:
            for n in range(N_node):
                for r in range(R):
                    if x[self.node_power_var_index(n, r)] == 1:
                        cost += self.microgrid.power_states[r].energy_cost_tariff
        else:
            # R == 1: y_{n,0} = 1 is fixed for all nodes
            cost += N_node * self.microgrid.power_states[0].energy_cost_tariff

        return round(float(cost), 6)

    def variable_labels(self) -> list[str]:
        labels = [
            f"x_T{m}_N{n}"
            for m in range(self.num_tasks)
            for n in range(self.num_nodes)
        ]
        if self.has_mode_variables:
            for n in range(self.num_nodes):
                for r in range(self.num_power_states):
                    labels.append(f"y_N{n}_P{r}")
        return labels


def generate_coupled_problem(
    num_nodes: int = 3,
    num_tasks: int = 4,
    num_power_states: int = 2,
    seed: int = 42,
    name: str | None = None,
    capacity_scale: float = 1.0,
) -> CoupledInfrastructureProblem:
    """
    Synthesize a deterministic Coupled Routing & Power-Mode Selection problem.
    Setting `capacity_scale < 0.5` deliberately creates overloaded/infeasible instances
    for testing validator rejection.
    """
    rng = np.random.default_rng(seed)
    accel_profiles = [
        ("NVIDIA-Jetson-Orin-AGX", 1.8, 135.0, 22.0),
        ("NVIDIA-L40S-Edge", 1.1, 190.0, 40.0),
        ("AMD-MI300A-MicroNode", 0.9, 220.0, 50.0),
        ("Qualcomm-Cloud-AI-100", 2.2, 115.0, 15.0),
    ]

    nodes: list[EdgeComputeNode] = []
    for n in range(num_nodes):
        arch, base_lat, cap_w, idle_w = accel_profiles[n % len(accel_profiles)]
        jitter = float(rng.uniform(0.92, 1.08))
        nodes.append(
            EdgeComputeNode(
                node_id=n,
                name=f"EdgeNode-{n}-{arch.split('-')[1]}",
                accelerator_type=arch,
                base_latency_ms=round(base_lat * jitter, 3),
                thermal_capacity_watts=round(cap_w * jitter * capacity_scale, 2),
                idle_power_watts=round(idle_w, 2),
                electrical_phase=n % 3,
            )
        )

    dist_mat = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            d = round(float(rng.uniform(0.8, 3.2)), 3)
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    tasks: list[InferenceBatchTask] = []
    for m in range(num_tasks):
        flops = round(float(rng.uniform(25.0, 85.0)), 2)
        demand_w = round(float(rng.uniform(40.0, 80.0)), 2)
        prio = round(float(rng.uniform(0.8, 1.5)), 2)
        tasks.append(
            InferenceBatchTask(
                task_id=m,
                name=f"LLM-Batch-{m}",
                compute_flops_giga=flops,
                thermal_load_watts=demand_w,
                priority_weight=prio,
                deadline_ms=200.0,
            )
        )

    sync_mat = np.zeros((num_tasks, num_tasks), dtype=np.float64)
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            if rng.random() < 0.65:
                w = round(float(rng.uniform(0.4, 2.0)), 3)
                sync_mat[i, j] = w
                sync_mat[j, i] = w

    power_states: list[MicrogridPowerState] = []
    if num_power_states == 1:
        power_states.append(
            MicrogridPowerState(
                state_id=0,
                label="Nominal_Envelope",
                energy_cost_tariff=1.0,
                cooling_multiplier=1.0,
                is_eco_throttled=False,
            )
        )
    else:
        for r in range(num_power_states):
            if r == 0:
                power_states.append(
                    MicrogridPowerState(
                        state_id=0,
                        label="Solar_BESS_Eco",
                        energy_cost_tariff=-1.2,
                        cooling_multiplier=0.85,
                        is_eco_throttled=True,
                    )
                )
            elif r == 1:
                power_states.append(
                    MicrogridPowerState(
                        state_id=1,
                        label="Grid_Active_HighCap",
                        energy_cost_tariff=2.4,
                        cooling_multiplier=1.35,
                        is_eco_throttled=False,
                    )
                )
            else:
                power_states.append(
                    MicrogridPowerState(
                        state_id=r,
                        label=f"Mode_{r}",
                        energy_cost_tariff=round(1.5 * r, 2),
                        cooling_multiplier=round(1.0 + 0.25 * r, 2),
                        is_eco_throttled=False,
                    )
                )

    problem_name = (
        name or f"PravahaQ-{num_nodes}N-{num_tasks}T-{num_power_states}P-Seed{seed}"
    )
    return CoupledInfrastructureProblem(
        name=problem_name,
        cluster=EdgeClusterTopology(
            nodes=tuple(nodes),
            interconnect_latency_ms=tuple(tuple(float(v) for v in row) for row in dist_mat),
        ),
        workload=WorkloadGraph(
            tasks=tuple(tasks),
            kv_sync_weights=tuple(tuple(float(v) for v in row) for row in sync_mat),
        ),
        microgrid=MicrogridSlice(
            power_states=tuple(power_states),
            solar_generation_kw=48.0,
            battery_soc_fraction=0.75,
            phase_overload_penalty=5.5,
        ),
    )
