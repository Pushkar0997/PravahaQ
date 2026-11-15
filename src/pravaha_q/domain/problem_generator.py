"""Deterministic generator for Coupled Edge-AI & Microgrid optimization instances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pravaha_q.domain.cluster_topology import EdgeClusterTopology, EdgeComputeNode
from pravaha_q.domain.microgrid_physics import MicrogridPowerState, MicrogridSlice
from pravaha_q.domain.workload_trace import InferenceBatchTask, WorkloadGraph


@dataclass(frozen=True)
class CoupledInfrastructureProblem:
    """
    Complete Coupled Edge-AI Routing + Microgrid Thermal Dispatch instance.

    Binary variable layout x in {0, 1}^N:
      - First M * N_node variables: x_{m, n} = 1 if task m is routed to node n.
      - Next N_node * R variables:  y_{n, r} = 1 if node n operates in power state r.
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
    def num_routing_vars(self) -> int:
        return self.num_tasks * self.num_nodes

    @property
    def num_power_vars(self) -> int:
        return self.num_nodes * self.num_power_states

    @property
    def total_binary_variables(self) -> int:
        return self.num_routing_vars + self.num_power_vars

    def task_node_var_index(self, task_m: int, node_n: int) -> int:
        """Return flat binary variable index for routing task_m -> node_n."""
        return task_m * self.num_nodes + node_n

    def node_power_var_index(self, node_n: int, state_r: int) -> int:
        """Return flat binary variable index for node_n operating in power state_r."""
        return self.num_routing_vars + node_n * self.num_power_states + state_r

    def variable_labels(self) -> list[str]:
        """Return human-readable labels for all N binary variables."""
        labels: list[str] = []
        for m in range(self.num_tasks):
            for n in range(self.num_nodes):
                labels.append(f"x_T{m}_N{n}")
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
) -> CoupledInfrastructureProblem:
    """
    Synthesize a realistic, deterministic coupled Edge-AI + 3-Phase Microgrid problem.
    """
    rng = np.random.default_rng(seed)
    accel_profiles = [
        ("NVIDIA-Jetson-Orin-AGX", 1.8, 120.0, 22.0),
        ("NVIDIA-L40S-Edge", 1.1, 180.0, 40.0),
        ("AMD-MI300A-MicroNode", 0.9, 210.0, 50.0),
        ("Qualcomm-Cloud-AI-100", 2.2, 95.0, 15.0),
    ]

    nodes: list[EdgeComputeNode] = []
    for n in range(num_nodes):
        arch, base_lat, therm_cap, idle_w = accel_profiles[n % len(accel_profiles)]
        jitter = float(rng.uniform(0.9, 1.1))
        nodes.append(
            EdgeComputeNode(
                node_id=n,
                name=f"EdgeNode-{n}-{arch.split('-')[1]}",
                accelerator_type=arch,
                base_latency_ms=round(base_lat * jitter, 3),
                thermal_capacity_watts=round(therm_cap * jitter, 2),
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
        flops = round(float(rng.uniform(25.0, 90.0)), 2)
        heat = round(float(rng.uniform(45.0, 95.0)), 2)
        prio = round(float(rng.uniform(0.8, 1.5)), 2)
        tasks.append(
            InferenceBatchTask(
                task_id=m,
                name=f"LLM-Batch-{m}",
                compute_flops_giga=flops,
                thermal_load_watts=heat,
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
    for r in range(num_power_states):
        if r == 0:
            power_states.append(
                MicrogridPowerState(
                    state_id=0,
                    label="Solar_BESS_Eco",
                    energy_cost_tariff=-1.8,
                    cooling_multiplier=0.82,
                    is_eco_throttled=True,
                )
            )
        elif r == 1:
            power_states.append(
                MicrogridPowerState(
                    state_id=1,
                    label="Grid_Active_Cool",
                    energy_cost_tariff=2.4,
                    cooling_multiplier=1.35,
                    is_eco_throttled=False,
                )
            )
        else:
            power_states.append(
                MicrogridPowerState(
                    state_id=r,
                    label=f"Dispatch_Level_{r}",
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
