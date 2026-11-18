"""Strict Pydantic v2 Request and Response Contracts for the PravahaQ Control Plane."""

from __future__ import annotations

from pydantic import BaseModel, Field

from pravaha_q.infrastructure.admission_guard import AlgorithmType, BackendType


class ProblemSpecRequest(BaseModel):
    """Parameters defining a Coupled Edge-AI + 3-Phase Microgrid problem instance."""

    num_nodes: int = Field(default=2, ge=1, le=8)
    num_tasks: int = Field(default=3, ge=1, le=16)
    num_power_states: int = Field(default=2, ge=1, le=4)
    capacity_scale: float = Field(
        default=1.0,
        gt=0.05,
        le=5.0,
        description="Multiplier on node capacity envelopes; <0.3 creates overloaded test instances.",
    )
    enable_cubic_3body: bool = False
    unbalanced_lambda1: float = Field(default=3.5, gt=0.0)
    unbalanced_lambda2: float = Field(default=6.0, gt=0.0)
    equality_penalty: float = Field(default=10.0, gt=0.0)
    seed: int = 42


class CompileRequest(ProblemSpecRequest):
    """Request payload for HUBO/Ising compilation and OpenQASM 3.0 / CUDA-Q export."""

    topology: str = Field(default="heavy_hex")
    gamma: float = 0.45
    beta: float = 0.30
    include_counterdiabatic: bool = True


class OptimizeRequest(ProblemSpecRequest):
    """Request payload for executing a hybrid quantum-classical optimization job."""

    algorithm: AlgorithmType = Field(default=AlgorithmType.BF_DCQO)
    sdk: BackendType = Field(default=BackendType.QISKIT)
    pce_qubits: int = Field(default=5, ge=3, le=12)
    mps_bond_dim: int = Field(default=32, ge=2, le=64)
    bf_iterations: int = Field(default=4, ge=1, le=10)
    shots: int = Field(default=2048, ge=128, le=16384)
    cvar_alpha: float = Field(default=0.20, gt=0.01, le=1.0)
    enable_zne: bool = True
    enable_trex: bool = True


class OptimizeSchedulePayload(BaseModel):
    """Independently validated schedule and minimum-Hamming MILP repair payload."""

    raw_bitstring: str
    repaired_bitstring: str
    was_repaired: bool
    is_feasible: bool
    validity: str
    solver_status: str
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


class OptimizeResponse(BaseModel):
    """Complete optimization, admission, validation, and QEM response."""

    job_id: str
    problem_name: str
    hamiltonian_sha256: str
    num_binary_variables: int
    num_qubits_executed: int
    estimated_memory_mb: float
    algorithm: str
    sdk: str
    execution_mode: str
    operational_cost_f: float | None
    best_energy: float
    exact_ground_energy: float | None
    approximation_ratio: float | None
    cvar_expectation_energy: float
    zne_extrapolated_energy: float
    zne_std_error: float
    execution_time_ms: float
    bitstring_counts: dict[str, int]
    schedule: OptimizeSchedulePayload
    iteration_telemetry: list[dict[str, float | int | list[float]]]
