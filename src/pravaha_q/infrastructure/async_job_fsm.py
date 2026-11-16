"""Finite State Machine (FSM) for Hybrid Quantum-Classical Job Lifecycle Management."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class JobState(str, Enum):
    """Canonical lifecycle states of a PravahaQ optimization job."""

    QUEUED = "QUEUED"
    COMPILING_HUBO = "COMPILING_HUBO"
    EXECUTING_QUANTUM = "EXECUTING_QUANTUM"
    MITIGATING_ERRORS = "MITIGATING_ERRORS"
    REPAIRING_FEASIBILITY = "REPAIRING_FEASIBILITY"
    COMPLETED = "COMPLETED"
    FAILED_FALLBACK_MPS = "FAILED_FALLBACK_MPS"


@dataclass
class QuantumJobRecord:
    """Tracks state transitions and timestamps for an optimization request."""

    job_id: str = field(default_factory=lambda: f"pq-{uuid.uuid4().hex[:10]}")
    algorithm: str = "bf_dcqo"
    sdk: str = "qiskit"
    state: JobState = JobState.QUEUED
    state_history: list[tuple[str, float]] = field(default_factory=list)
    result_bitstring: str | None = None
    result_energy: float | None = None

    def transition_to(self, new_state: JobState) -> None:
        self.state = new_state
        self.state_history.append((new_state.value, round(time.time(), 4)))
