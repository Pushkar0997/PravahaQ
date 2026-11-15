"""Domain entities and infrastructure digital twin for PravahaQ."""

from pravaha_q.domain.cluster_topology import EdgeClusterTopology, EdgeComputeNode
from pravaha_q.domain.exceptions import (
    FeasibilityRepairError,
    HUBOCompilationError,
    PravahaError,
    QubitCompressionCapacityError,
    TopologyValidationError,
)
from pravaha_q.domain.microgrid_physics import MicrogridPowerState, MicrogridSlice
from pravaha_q.domain.problem_generator import (
    CoupledInfrastructureProblem,
    generate_coupled_problem,
)
from pravaha_q.domain.workload_trace import InferenceBatchTask, WorkloadGraph

__all__ = [
    "CoupledInfrastructureProblem",
    "EdgeClusterTopology",
    "EdgeComputeNode",
    "FeasibilityRepairError",
    "HUBOCompilationError",
    "InferenceBatchTask",
    "MicrogridPowerState",
    "MicrogridSlice",
    "PravahaError",
    "QubitCompressionCapacityError",
    "TopologyValidationError",
    "WorkloadGraph",
    "generate_coupled_problem",
]
