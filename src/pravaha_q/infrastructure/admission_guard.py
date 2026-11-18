"""
Algorithm-Specific Resource, Memory & Compatibility Admission Guard (Step 8)
============================================================================
Enforces operational safeguards before any quantum or classical solver executes:
  1. Strict enumeration of supported `AlgorithmType` and `BackendType` identifiers.
  2. Algorithm-to-backend compatibility matrix validation.
  3. Algorithm-specific memory and state-space admission rules:
     - Dense statevector (`bf_dcqo`, `cd_warm_qaoa`, `iqo_bnb`, `exact`):
       Requires 2^N complex128 amplitudes + working buffers; capped at N <= 18 qubits.
     - Multistage PCE (`pce`):
       Requires 2^{n_pce} amplitudes + N correlator bitmasks; capped at n_pce <= 12, N <= 200.
     - Matrix Product State (`mps`):
       Requires O(N * chi^2) tensor storage; capped at N <= 50, chi <= 64.
     - Structured Enumeration (`structured_enum`):
       Requires N_node^M * R^{N_node} <= 200,000 combinations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pravaha_q.domain.exceptions import PravahaError
from pravaha_q.domain.problem_generator import CoupledInfrastructureProblem


class AlgorithmType(str, Enum):
    """Enumerated optimization algorithms supported by PravahaQ."""

    BF_DCQO = "bf_dcqo"
    PCE = "pce"
    IQO_BNB = "iqo_bnb"
    CD_WARM_QAOA = "cd_warm_qaoa"
    MPS = "mps"
    EXACT = "exact"
    STRUCTURED_ENUM = "structured_enum"
    GREEDY = "greedy"
    SIMULATED_ANNEALING = "simulated_annealing"


class BackendType(str, Enum):
    """Enumerated execution backends supported by PravahaQ."""

    QISKIT = "qiskit"
    PENNYLANE = "pennylane"
    CIRQ = "cirq"
    MPS = "mps"
    IBM_RUNTIME = "ibm_runtime"
    CLASSICAL = "classical"


class ResourceAdmissionError(PravahaError):
    """Raised when a problem or solver configuration exceeds memory or qubit budgets."""


class CompatibilityError(PravahaError):
    """Raised when an incompatible (algorithm, backend) pair is requested."""


_COMPATIBLE_BACKENDS: dict[AlgorithmType, set[BackendType]] = {
    AlgorithmType.BF_DCQO: {
        BackendType.QISKIT,
        BackendType.CIRQ,
        BackendType.IBM_RUNTIME,
    },
    AlgorithmType.PCE: {
        BackendType.PENNYLANE,
        BackendType.QISKIT,
    },
    AlgorithmType.IQO_BNB: {
        BackendType.QISKIT,
        BackendType.CLASSICAL,
    },
    AlgorithmType.CD_WARM_QAOA: {
        BackendType.QISKIT,
        BackendType.PENNYLANE,
    },
    AlgorithmType.MPS: {
        BackendType.MPS,
        BackendType.CLASSICAL,
    },
    AlgorithmType.EXACT: {
        BackendType.CLASSICAL,
    },
    AlgorithmType.STRUCTURED_ENUM: {
        BackendType.CLASSICAL,
    },
    AlgorithmType.GREEDY: {
        BackendType.CLASSICAL,
    },
    AlgorithmType.SIMULATED_ANNEALING: {
        BackendType.CLASSICAL,
    },
}


@dataclass(frozen=True)
class ResourceBudget:
    """Hard admission ceilings for pre-execution safety checks."""

    max_dense_statevector_qubits: int = 18
    max_pce_qubits: int = 12
    max_pce_variables: int = 200
    max_mps_qubits: int = 50
    max_mps_bond_dim: int = 64
    max_structured_enum_states: int = 200_000
    max_memory_mb: float = 512.0
    max_wall_clock_budget_ms: float = 5000.0


@dataclass(frozen=True)
class AdmissionDecision:
    """Audit record returned by the pre-execution resource admission guard."""

    admitted: bool
    algorithm: AlgorithmType
    backend: BackendType
    num_binary_variables: int
    num_qubits_allocated: int
    estimated_memory_mb: float
    reason: str = "ADMITTED"


def estimate_solver_memory_mb(
    algorithm: AlgorithmType,
    num_variables: int,
    pce_qubits: int = 6,
    mps_bond_dim: int = 32,
) -> tuple[int, float]:
    """
    Return `(qubits_allocated, estimated_memory_mb)` for a given solver regime.
    """
    if algorithm == AlgorithmType.PCE:
        n_q = min(pce_qubits, num_variables)
        if n_q > 30:
            return n_q, float("inf")
        dim = 1 << n_q
        # Statevector (16B) + N correlator phase masks (16B * N * 2^n_q)
        bytes_needed = dim * 16 * (2 + num_variables)
        return n_q, round(bytes_needed / (1024.0 * 1024.0), 4)

    if algorithm == AlgorithmType.MPS:
        n_q = num_variables
        # N rank-3 complex128 tensors of shape (chi, 2, chi) + SVD workspace
        bytes_needed = n_q * 2 * (mps_bond_dim**2) * 16 * 4
        return n_q, round(max(bytes_needed / (1024.0 * 1024.0), 0.05), 4)

    if algorithm in (
        AlgorithmType.STRUCTURED_ENUM,
        AlgorithmType.GREEDY,
        AlgorithmType.SIMULATED_ANNEALING,
    ):
        return num_variables, round(0.02 * num_variables, 4)

    # Dense statevector algorithms: BF_DCQO, CD_WARM_QAOA, IQO_BNB, EXACT
    n_q = num_variables
    if n_q > 30:
        return n_q, float("inf")
    dim = 1 << n_q
    # Statevector (16B) + diagonal Hamiltonian (8B) + working buffer (32B) = 56 bytes per state
    bytes_needed = dim * 56
    return n_q, round(bytes_needed / (1024.0 * 1024.0), 4)


def check_execution_admission(
    problem: CoupledInfrastructureProblem,
    algorithm: str | AlgorithmType,
    backend: str | BackendType,
    pce_qubits: int = 6,
    mps_bond_dim: int = 32,
    budget: ResourceBudget | None = None,
) -> AdmissionDecision:
    """
    Validate algorithm/backend enums, compatibility, and memory/qubit ceilings.
    Raises `CompatibilityError` or `ResourceAdmissionError` if rejected.
    """
    limits = budget or ResourceBudget()

    try:
        algo_enum = (
            algorithm
            if isinstance(algorithm, AlgorithmType)
            else AlgorithmType(str(algorithm).lower().strip())
        )
    except ValueError as exc:
        valid = [a.value for a in AlgorithmType]
        raise CompatibilityError(
            f"Unsupported algorithm '{algorithm}'. Valid options: {valid}"
        ) from exc

    try:
        backend_enum = (
            backend
            if isinstance(backend, BackendType)
            else BackendType(str(backend).lower().strip())
        )
    except ValueError as exc:
        valid_b = [b.value for b in BackendType]
        raise CompatibilityError(
            f"Unsupported backend '{backend}'. Valid options: {valid_b}"
        ) from exc

    allowed_backends = _COMPATIBLE_BACKENDS.get(algo_enum, set())
    if backend_enum not in allowed_backends:
        raise CompatibilityError(
            f"Algorithm '{algo_enum.value}' is incompatible with backend '{backend_enum.value}'. "
            f"Compatible backends: {sorted(b.value for b in allowed_backends)}"
        )

    n_vars = problem.total_binary_variables
    q_alloc, est_mem_mb = estimate_solver_memory_mb(
        algo_enum,
        num_variables=n_vars,
        pce_qubits=pce_qubits,
        mps_bond_dim=mps_bond_dim,
    )

    if algo_enum in (
        AlgorithmType.BF_DCQO,
        AlgorithmType.CD_WARM_QAOA,
        AlgorithmType.IQO_BNB,
        AlgorithmType.EXACT,
    ):
        if n_vars > limits.max_dense_statevector_qubits:
            raise ResourceAdmissionError(
                f"Dense statevector algorithm '{algo_enum.value}' requested for N={n_vars} variables, "
                f"exceeding max_dense_statevector_qubits={limits.max_dense_statevector_qubits}. "
                "Use 'pce', 'mps', or 'greedy' for larger instances."
            )

    if algo_enum == AlgorithmType.PCE:
        if pce_qubits > limits.max_pce_qubits:
            raise ResourceAdmissionError(
                f"PCE requested with pce_qubits={pce_qubits} > max_pce_qubits={limits.max_pce_qubits}."
            )
        if n_vars > limits.max_pce_variables:
            raise ResourceAdmissionError(
                f"PCE requested for N={n_vars} > max_pce_variables={limits.max_pce_variables}."
            )

    if algo_enum == AlgorithmType.MPS:
        if n_vars > limits.max_mps_qubits:
            raise ResourceAdmissionError(
                f"MPS requested for N={n_vars} > max_mps_qubits={limits.max_mps_qubits}."
            )
        if mps_bond_dim > limits.max_mps_bond_dim:
            raise ResourceAdmissionError(
                f"MPS bond dimension {mps_bond_dim} > max_mps_bond_dim={limits.max_mps_bond_dim}."
            )

    if algo_enum == AlgorithmType.STRUCTURED_ENUM:
        r_eff = problem.num_power_states if problem.has_mode_variables else 1
        combos = (problem.num_nodes**problem.num_tasks) * (
            r_eff**problem.num_nodes
        )
        if combos > limits.max_structured_enum_states:
            raise ResourceAdmissionError(
                f"Structured enumeration space {combos} exceeds limit {limits.max_structured_enum_states}."
            )

    if est_mem_mb > limits.max_memory_mb:
        raise ResourceAdmissionError(
            f"Estimated solver memory {est_mem_mb:.2f} MB exceeds budget {limits.max_memory_mb:.2f} MB."
        )

    return AdmissionDecision(
        admitted=True,
        algorithm=algo_enum,
        backend=backend_enum,
        num_binary_variables=n_vars,
        num_qubits_allocated=q_alloc,
        estimated_memory_mb=est_mem_mb,
        reason="ADMITTED",
    )
