# ADR-0002: Bias-Field Digitized Counterdiabatic Quantum Optimization (`BF-DCQO`) as Primary Solver over Variational QAOA

* **Status:** Accepted
* **Date:** 2026-11-16
* **Author:** Pushkar Kumar

## Context
Standard Variational QAOA requires an outer classical optimization loop (`COBYLA`, `SPSA`, `Adam`) executing 200 to 1,000 circuit evaluations to tune $(\boldsymbol{\gamma}, \boldsymbol{\beta})$. Over cloud QPU queues, this introduces minutes to hours of latency and suffers from exponentially vanishing gradients (Barren Plateaus) and local traps at shallow depth $p \le 2$.

## Decision
We adopt **Bias-Field Digitized Counterdiabatic Quantum Optimization (`BF-DCQO`)** as the primary direct-regime solver in `PravahaQ`, while retaining `CD-WarmQAOA` for comparative benchmarking:
1. Gauge coefficients $\alpha_1^*(\lambda)$ for the nested commutator Adiabatic Gauge Potential $\hat{\mathcal{A}}_\lambda^{(1)} = i\alpha_1[\hat{H}_{\text{ad}}, \partial_\lambda \hat{H}_{\text{ad}}]$ are computed analytically in closed form ($O(\vert{}E\vert{} + \vert{}T\vert{})$ time).
2. Instead of optimizing variational angles, we execute $K \in [3, 5]$ fixed-schedule counterdiabatic impulses and iteratively update the longitudinal bias fields $h_i^{(k+1)} = h_i^{(0)} + \eta_k \tanh(\gamma \langle \hat{Z}_i \rangle^{(k)})$.

## Consequences
* **Positive:** Reduces total QPU circuit executions by **50x–150x** (from ~400 iterations to 4 iterations) while achieving higher Approximation Ratios ($AR > 0.94$) at depth $p=1$.
* **Trade-off:** Requires evaluating single-qubit magnetizations $\langle \hat{Z}_i \rangle$ between bias-field steps.
