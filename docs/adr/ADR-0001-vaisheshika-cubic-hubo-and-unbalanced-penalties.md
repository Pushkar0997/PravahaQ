# ADR-0001: Adoption of 3-Local Cubic HUBO (*Vaisheshika* Compiler) and Slack-Free Unbalanced Inequality Penalization

* **Status:** Accepted
* **Date:** 2026-11-16
* **Author:** Pushkar Kumar (Principal Architect, PravahaQ)

## Context
Coupled Edge-AI inference routing and 3-phase microgrid thermal dispatch exhibits two structural characteristics that break standard Quadratic Unconstrained Binary Optimization (QUBO) compilers:
1. **Multi-Point Non-Linear Congestion:** Simultaneous execution of two high-TDP AI workloads on nodes sharing a transformer phase under low cooling headroom produces a 3-body non-linear thermal spike ($x_i x_j x_k$). Standard Rosenberg quadratization replaces $x_i x_j$ with an auxiliary binary variable $y_{ij}$ plus a large penalty $M(x_i x_j - 2x_i y_{ij} - 2x_j y_{ij} + 3y_{ij})$, increasing qubit count and dynamic range ($\max |J_{ij}| / \min |J_{ij}|$), which degrades QPU gate resolution.
2. **Inequality Capacity Constraints:** Standard slack-variable encoding of $M$ inequality constraints with capacity $C$ adds $M \lceil \log_2 C \rceil$ ancilla qubits.

## Decision
1. We formulate the native cost functional as a **3-Local Cubic HUBO** ($E(\mathbf{x}) = \sum L_i x_i + \sum Q_{ij} x_i x_j + \sum T_{ijk} x_i x_j x_k$) and map it directly to a 3-local Ising Hamiltonian $\hat{H}_C$ decomposed via 4-CX staircases with Paulihedral prefix cancellation.
2. We encode all thermal and power inequality constraints $g_n(\mathbf{x}) \ge 0$ using **Montañez-Barrera Unbalanced Penalization** ($\Phi(g) = -\lambda_1 g(\mathbf{x}) + \lambda_2 g(\mathbf{x})^2$), requiring **zero slack qubits**.

## Consequences
* **Positive:** Reduces physical qubit count by **42%–58%** across benchmark instances and keeps Hamiltonian spectral dynamic range $4.5\times$ narrower than Rosenberg quadratization.
* **Trade-off:** Introduces 3-body $\hat{Z}_i \hat{Z}_j \hat{Z}_k$ terms requiring 4 `CX` gates each (mitigated via lexicographical Paulihedral gate cancellation).
