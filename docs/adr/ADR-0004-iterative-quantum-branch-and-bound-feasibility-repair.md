# ADR-0004: Iterative Quantum Branch-and-Bound (`IQO-BnB`) and Deterministic Post-Quantum Feasibility Repair

* **Status:** Accepted
* **Date:** 2026-11-16
* **Author:** Pushkar Kumar

## Context
1. Standard Recursive QAOA (RQAOA) greedily freezes the single spin pair $(i^*, j^*)$ with maximal $|\langle \hat{Z}_i \hat{Z}_j \rangle|$. On frustrated 3-body spin glasses, a single early greedy mistake locks out the global ground state permanently.
2. Furthermore, raw bitstrings sampled from noisy hardware can violate hard single-assignment or thermal ceiling constraints.

## Decision
1. We replace greedy RQAOA with **Backtracking Iterative Quantum Branch-and-Bound (`IQO-BnB`)**, which explores a bounded depth-$d$ branching tree over top-correlated spin pairs and prunes branches using continuous LP/QP lower bounds.
2. Every solver output must pass through `FeasibilityRepairEngine` (`src/pravaha_q/math_engine/feasibility_repair.py`), which deterministically repairs any unassigned/multi-assigned tasks and resolves thermal cap breaches via greedy marginal-cost swaps in $O(M \cdot N_{\text{node}})$ time.

## Consequences
* **Positive:** Guarantees **100.0% constraint feasibility** on every API call regardless of QPU noise level, making PravahaQ safe for production infrastructure control.
