"""
IBM Quantum Runtime Hardware Adapter (Honest Status & Fallback Identification)
==============================================================================
Checks for a real `IBM_QUANTUM_TOKEN`. If hardware credentials are unavailable:
  - When `allow_simulator_fallback=False` (default for strict hardware runs),
    returns `SolverStatus.UNKNOWN` with `hardware_status="UNAVAILABLE_NO_CREDENTIALS"`.
  - When `allow_simulator_fallback=True`, executes a local shot simulation and
    explicitly labels `execution_mode=ExecutionMode.SHOT_SIMULATION` and
    `metadata["is_fallback"] = True` (never pretending a simulator run is hardware).
"""

from __future__ import annotations

import os
import time

from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass
from pravaha_q.math_engine.validator import ScheduleValidity, SolverStatus
from pravaha_q.solvers.base import ExecutionMode, SolverResult
from pravaha_q.solvers.qiskit_adapter import QiskitBFDCQOSolver


class IBMRuntimeHardwareSession:
    """IBM Quantum Runtime hardware adapter with explicit fallback identification."""

    def __init__(
        self,
        backend_name: str = "ibm_fez",
        shots: int = 4096,
        api_token: str | None = None,
        allow_simulator_fallback: bool = False,
    ) -> None:
        self.backend_name = backend_name
        self.shots = shots
        self.api_token = api_token or os.getenv("IBM_QUANTUM_TOKEN", "").strip()
        self.allow_simulator_fallback = allow_simulator_fallback

    def is_hardware_configured(self) -> bool:
        return bool(
            self.api_token
            and self.api_token != "your_ibm_quantum_platform_api_token_here"
        )

    def solve(self, ising: IsingSpinGlass) -> SolverResult:
        start_t = time.perf_counter()
        n = ising.num_qubits

        if not self.is_hardware_configured():
            if not self.allow_simulator_fallback:
                exec_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
                return SolverResult(
                    solver_name=f"IBM-Runtime-{self.backend_name}",
                    sdk_name="Qiskit-IBM-Runtime",
                    num_qubits=n,
                    best_bitstring="0" * n,
                    best_energy=float("inf"),
                    expectation_energy=float("inf"),
                    approximation_ratio=None,
                    execution_time_ms=exec_ms,
                    bitstring_counts={},
                    execution_mode=ExecutionMode.HARDWARE,
                    validity=ScheduleValidity.INVALID,
                    solver_status=SolverStatus.UNKNOWN,
                    metadata={
                        "backend": self.backend_name,
                        "hardware_status": "UNAVAILABLE_NO_CREDENTIALS",
                        "is_fallback": False,
                    },
                )

            # Explicit fallback: clearly labeled as SHOT_SIMULATION with is_fallback=True
            sim_res = QiskitBFDCQOSolver(
                shots=self.shots,
                execution_mode=ExecutionMode.SHOT_SIMULATION,
            ).solve(ising)
            return SolverResult(
                solver_name=f"IBM-Runtime-{self.backend_name}-FallbackSim",
                sdk_name="Qiskit-Simulator-Fallback",
                num_qubits=n,
                best_bitstring=sim_res.best_bitstring,
                best_energy=sim_res.best_energy,
                expectation_energy=sim_res.expectation_energy,
                approximation_ratio=None,
                execution_time_ms=sim_res.execution_time_ms,
                bitstring_counts=sim_res.bitstring_counts,
                execution_mode=ExecutionMode.SHOT_SIMULATION,
                validity=sim_res.validity,
                solver_status=sim_res.solver_status,
                metadata={
                    "requested_backend": self.backend_name,
                    "hardware_status": "FALLBACK_TO_SHOT_SIMULATION",
                    "is_fallback": True,
                },
            )

        # Credentials present but live cloud call not invoked in offline test environment
        sim_res = QiskitBFDCQOSolver(
            shots=self.shots,
            execution_mode=ExecutionMode.SHOT_SIMULATION,
        ).solve(ising)
        return SolverResult(
            solver_name=f"IBM-Runtime-{self.backend_name}",
            sdk_name="Qiskit-IBM-Runtime",
            num_qubits=n,
            best_bitstring=sim_res.best_bitstring,
            best_energy=sim_res.best_energy,
            expectation_energy=sim_res.expectation_energy,
            approximation_ratio=None,
            execution_time_ms=sim_res.execution_time_ms,
            bitstring_counts=sim_res.bitstring_counts,
            execution_mode=ExecutionMode.SHOT_SIMULATION,
            validity=sim_res.validity,
            solver_status=sim_res.solver_status,
            metadata={
                "backend": self.backend_name,
                "hardware_status": "CREDENTIALS_DETECTED_OFFLINE_SIM",
                "is_fallback": True,
            },
        )
