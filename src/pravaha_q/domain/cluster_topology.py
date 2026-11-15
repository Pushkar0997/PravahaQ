"""Edge-AI compute cluster node and interconnect topology models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pravaha_q.domain.exceptions import TopologyValidationError


@dataclass(frozen=True)
class EdgeComputeNode:
    """Represents a physical Edge-AI compute node (e.g., NVIDIA H100 or Jetson Orin)."""

    node_id: int
    name: str
    accelerator_type: str
    base_latency_ms: float
    thermal_capacity_watts: float
    idle_power_watts: float
    electrical_phase: int = 0  # 0 = Phase A, 1 = Phase B, 2 = Phase C

    def __post_init__(self) -> None:
        if self.thermal_capacity_watts <= 0:
            raise TopologyValidationError(
                f"Node {self.name} must have positive thermal_capacity_watts."
            )
        if self.electrical_phase not in (0, 1, 2):
            raise TopologyValidationError(
                f"Node {self.name} electrical_phase must be 0, 1, or 2."
            )


@dataclass(frozen=True)
class EdgeClusterTopology:
    """Represents a distributed Edge-AI cluster with inter-node network distances."""

    nodes: tuple[EdgeComputeNode, ...]
    interconnect_latency_ms: tuple[tuple[float, ...], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.nodes) == 0:
            raise TopologyValidationError("EdgeClusterTopology requires at least 1 node.")

        n = len(self.nodes)
        if self.interconnect_latency_ms and (
            len(self.interconnect_latency_ms) != n
            or any(len(row) != n for row in self.interconnect_latency_ms)
        ):
            raise TopologyValidationError(
                f"interconnect_latency_ms must be an {n}x{n} symmetric matrix."
            )

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def distance(self, node_a: int, node_b: int) -> float:
        """Return inter-node network transfer delay in milliseconds."""
        if node_a == node_b:
            return 0.0
        if not self.interconnect_latency_ms:
            return 1.5
        return float(self.interconnect_latency_ms[node_a][node_b])

    def to_distance_matrix(self) -> np.ndarray:
        """Return the N_node x N_node numpy matrix of interconnect latencies."""
        n = self.num_nodes
        mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                mat[i, j] = self.distance(i, j)
        return mat
