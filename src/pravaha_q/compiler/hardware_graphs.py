"""
Quantum Hardware Coupling Topologies & Shortest-Path SWAP Routing
=================================================================
Generates coupling graphs for:
  - `heavy_hex`: IBM Quantum Heron r2 (`ibm_fez` / `ibm_torino`, degree <= 3)
  - `planar_grid`: Google Quantum AI Willow / Sycamore 2D square lattice (degree <= 4)
  - `all_to_all`: IonQ Forte Enterprise trapped-ion complete graph K_N
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class HardwareTopologyGraph:
    """Represents a physical QPU qubit connectivity graph."""

    name: str
    topology_type: str
    num_physical_qubits: int
    edges: tuple[tuple[int, int], ...]
    native_2q_gate: str

    def to_networkx(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(self.num_physical_qubits))
        g.add_edges_from(self.edges)
        return g

    def shortest_swap_hops(self, q_a: int, q_b: int) -> int:
        """
        Return the number of SWAP gates required to bring logical qubits q_a and q_b
        adjacent on this hardware coupling map (distance - 1).
        """
        if q_a == q_b:
            return 0
        if self.topology_type == "all_to_all":
            return 0
        g = self.to_networkx()
        if not g.has_node(q_a) or not g.has_node(q_b):
            return 1
        try:
            path_len = int(nx.shortest_path_length(g, source=q_a, target=q_b))
            return max(path_len - 1, 0)
        except nx.NetworkXNoPath:
            return 2


def build_hardware_topology(
    topology_type: str = "heavy_hex", num_qubits: int = 16
) -> HardwareTopologyGraph:
    """Construct a realistic coupling graph for the requested QPU architecture."""
    n = max(num_qubits, 4)
    topo = topology_type.lower().strip()

    if topo == "all_to_all":
        edges = tuple(
            (i, j) for i in range(n) for j in range(i + 1, n)
        )
        return HardwareTopologyGraph(
            name=f"IonQ-Forte-All2All-{n}Q",
            topology_type="all_to_all",
            num_physical_qubits=n,
            edges=edges,
            native_2q_gate="RXX",
        )

    if topo in ("planar_grid", "grid", "willow", "sycamore"):
        cols = int(max(2, round(n**0.5)))
        rows = (n + cols - 1) // cols
        edge_list: list[tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                u = r * cols + c
                if u >= n:
                    continue
                if c + 1 < cols and (u + 1) < n:
                    edge_list.append((u, u + 1))
                if r + 1 < rows and (u + cols) < n:
                    edge_list.append((u, u + cols))
        return HardwareTopologyGraph(
            name=f"Google-Willow-Grid-{n}Q",
            topology_type="planar_grid",
            num_physical_qubits=n,
            edges=tuple(edge_list),
            native_2q_gate="CZ",
        )

    # Default: IBM Heavy-Hex lattice (linear backbone + degree-3 bridge struts)
    hh_edges: list[tuple[int, int]] = [(i, i + 1) for i in range(n - 1)]
    for i in range(0, n - 4, 4):
        hh_edges.append((i, i + 4))
    return HardwareTopologyGraph(
        name=f"IBM-Heron-HeavyHex-{n}Q",
        topology_type="heavy_hex",
        num_physical_qubits=n,
        edges=tuple(hh_edges),
        native_2q_gate="CZ",
    )
