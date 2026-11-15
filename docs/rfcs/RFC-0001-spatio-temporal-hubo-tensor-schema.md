# RFC-0001: Spatio-Temporal HUBO & Ising Tensor Data Contract

* **RFC ID:** RFC-0001
* **Status:** Implemented (`v0.1.0`)
* **Target Module:** `src/pravaha_q/math_engine/`

## 1. Objective
Define a canonical, immutable tensor schema for representing up-to-3-body pseudo-Boolean optimization polynomials (HUBO) and their isomorphic Ising Spin-Glass Hamiltonians across all solvers, caches, and REST API payloads.

## 2. Canonical Index Ordering Invariant
All multi-qubit indices $(i, j)$ and $(i, j, k)$ MUST be stored in strictly ascending lexicographical order:
* **1-Body Linear / Longitudinal Field:** `dict[int, float]` where $0 \le i < N$.
* **2-Body Quadratic / ZZ Coupling:** `dict[tuple[int, int], float]` where $0 \le i < j < N$. Symmetric duplicates $(j, i)$ are merged into $(i, j)$ via $Q_{ij} \leftarrow Q_{ij} + Q_{ji}$ at construction time.
* **3-Body Cubic / ZZZ Coupling:** `dict[tuple[int, int, int], float]` where $0 \le i < j < k < N$. All $3! = 6$ permutations of $\{i, j, k\}$ are folded into the single canonical key $(i, j, k)$.

## 3. Spectral Normalization Contract
Before compiling gate rotation angles $(\gamma, \beta, \theta_{\text{CD}})$, the Ising Hamiltonian parameters $(h_i, J_{ij}, K_{ijk})$ are normalized by the spectral scale factor:
$$\Lambda_{\text{norm}} = \max \left( \max_i |h_i|, \, \max_{i<j} |J_{ij}|, \, \max_{i<j<k} |K_{ijk}|, \, 10^{-9} \right)$$
ensuring all dimensionless gate angles remain bounded within $[-\pi, \pi]$ and preventing modular $2\pi$ phase wrapping errors during Trotterization.
