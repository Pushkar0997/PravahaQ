# Changelog

All notable architectural and scientific changes to **PravahaQ** are documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-15 (Initial Frontier Release)

### Added
- **3-Local HUBO Formulation (`math_engine/hubo_builder.py`):** Explicit cubic $T_{ijk} x_i x_j x_k$ and $K_{ijk} \hat{Z}_i \hat{Z}_j \hat{Z}_k$ interactions modeling simultaneous compute-network-thermal phase congestion via the `Vaisheshika` compiler.
- **Slack-Free Unbalanced Inequality Penalization (`math_engine/unbalanced_penalties.py`):** Montañez-Barrera asymmetric parabolic-exponential encoding eliminating binary slack ancilla qubits.
- **Bias-Field Digitized Counterdiabatic Quantum Optimization (`algorithms/bf_dcqo.py`):** Analytical 1st/2nd-order nested commutator Adiabatic Gauge Potential ($\hat{\mathcal{A}}_\lambda$) + iterative longitudinal bias-field updates ($h_i^{(k+1)}$), eliminating outer-loop variational optimization.
- **Multistage Pauli Correlation Encoding (`encodings/pauli_correlation_encoding.py`):** Compresses $N \sim \mathcal{O}(n^k)$ binary scheduling variables onto $n \ll N$ physical qubits via $k$-body Pauli correlator expectation values.
- **Iterative Quantum Branch-and-Bound (`algorithms/iqo_branch_and_bound.py`):** Two-point spin correlation rounding $\langle \hat{Z}_i \hat{Z}_j \rangle$ with backtracking search and deterministic feasibility repair.
- **5-Backend Hybrid Execution Matrix:** Unified adapters for **Qiskit v2.x**, **PennyLane**, **Google Cirq**, **Matrix Product State (MPS) Tensor Networks**, and **OpenQASM 3.0 / NVIDIA CUDA-Q** code generation.
- **Quantum Error Mitigation Pipeline (`mitigation/`):** Zero-Noise Extrapolation (ZNE) gate folding, Twirled Readout Error Extinction (TREX) matrix inversion, and $\text{CVaR}_\alpha$ tail aggregation.
- **12 Interactive Jupyter Notebooks (`notebooks/`):** 6 Core Mathematical Deep-Dives + 6 Standalone Real-World Industry Mini-Projects.
- **Dual-Layer Documentation & Compiled Research Paper (`docs/`, `paper/`):** Non-technical analogies, worked 3-qubit numerical walkthroughs, Mermaid flowcharts, 6 ADRs, 3 RFCs, and the 8-figure research paper PDF.
