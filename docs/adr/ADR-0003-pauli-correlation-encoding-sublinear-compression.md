# ADR-0003: Multistage Pauli Correlation Encoding (`PCE`) for Sub-Linear Qubit Compression

* **Status:** Accepted
* **Date:** 2026-11-16
* **Author:** Pushkar Kumar

## Context
Enterprise edge clusters frequently involve $N \in [50, 200]$ binary decision variables, whereas local statevector simulators and near-term high-fidelity QPU subgraphs operate best at $n \in [6, 16]$ qubits. A strict 1-variable-per-qubit mapping cannot address $N > n$ without problem partitioning.

## Decision
We implement **Multistage Pauli Correlation Encoding (`PCE`)** in `src/pravaha_q/encodings/pauli_correlation_encoding.py`:
* Each binary spin variable $z_m \in \{-1, +1\}$ ($m = 1, \dots, N$) is mapped to the expectation value of a distinct $k$-body Pauli string $\hat{\Pi}_m \in \{I, X, Y, Z\}^{\otimes n}$ via $\tilde{z}_m(\boldsymbol{\theta}) = \tanh(\alpha_s \langle \psi(\boldsymbol{\theta}) | \hat{\Pi}_m | \psi(\boldsymbol{\theta}) \rangle)$.
* Using $k \in \{2, 3\}$ correlators on $n$ qubits yields capacity $N_{\max} = 3\binom{n}{2} + 3\binom{n}{3}$ (e.g., $n=7$ qubits encodes up to $63 + 105 = 168$ binary variables).

## Consequences
* **Positive:** Enables solving 50–150+ variable industrial scheduling instances on a 6- to 8-qubit circuit.
* **Trade-off:** Requires multistage annealing of the slope hyperparameter $\alpha_s$ and sign regularization $\beta_s$ using differentiable adjoint gradients (`PennyLane` / statevector backend).
