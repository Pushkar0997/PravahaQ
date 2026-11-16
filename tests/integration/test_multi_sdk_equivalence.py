"""Cross-SDK integration tests verifying Qiskit, PennyLane, Cirq, and Classical adapters."""

from __future__ import annotations

from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass
from pravaha_q.solvers.cirq_adapter import CirqGridSolver
from pravaha_q.solvers.classical_baselines import ExactSpectralSolver
from pravaha_q.solvers.pennylane_adapter import PennyLanePCESolver
from pravaha_q.solvers.qiskit_adapter import QiskitBFDCQOSolver


def test_all_sdk_adapters_solve_consistent_hamiltonian(
    small_ising: IsingSpinGlass,
) -> None:
    exact = ExactSpectralSolver().solve(small_ising)
    qk = QiskitBFDCQOSolver(shots=1024, seed=42).solve(small_ising)
    pl = PennyLanePCESolver(num_qubits=4, seed=42).solve(small_ising)
    cq = CirqGridSolver(seed=42).solve(small_ising)

    for res in (exact, qk, pl, cq):
        assert len(res.best_bitstring) == small_ising.num_qubits
        assert res.best_energy >= exact.best_energy - 1e-6
