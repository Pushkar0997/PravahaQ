# ADR-0006: Hexagonal Multi-SDK Solver Strategy and OpenQASM 3.0 / NVIDIA CUDA-Q Code Generation

* **Status:** Accepted
* **Date:** 2026-11-16
* **Author:** Pushkar Kumar

## Context
Quantum hardware providers use fragmented software stacks: IBM uses Qiskit v2 Primitives, Xanadu/AWS uses PennyLane, Google uses Cirq, and HPC supercomputing centers use NVIDIA CUDA-Q and Quantum Intermediate Representation (QIR / OpenQASM 3.0).

## Decision
All quantum solvers implement a strict `SolverStrategy` protocol (`src/pravaha_q/solvers/base.py`) consuming our SDK-agnostic `IsingSpinGlass` tensor object. In addition to live execution adapters for **Qiskit v2**, **PennyLane**, **Google Cirq**, and **MPS**, our compiler emits standalone **OpenQASM 3.0** programs and **NVIDIA CUDA-Q (`@cudaq.kernel`)** Python kernels.

## Consequences
* **Positive:** Zero vendor lock-in; mathematical parity across Qiskit, PennyLane, and Cirq is continuously enforced in CI (`tests/integration/test_multi_sdk_equivalence.py`) to within $10^{-7}$ statevector fidelity.
