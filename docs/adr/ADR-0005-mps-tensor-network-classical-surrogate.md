# ADR-0005: Matrix Product State (`MPS`) Tensor Network Classical Surrogate Engine

* **Status:** Accepted
* **Date:** 2026-11-16
* **Author:** Pushkar Kumar

## Context
Dense statevector simulation scales as $O(2^N)$ memory, hitting a hard ceiling around $N = 28$ qubits on standard servers. However, shallow $p=1, 2$ counterdiabatic circuits exhibit bounded bipartite entanglement entropy $S(\rho_A) \le O(p \log \chi)$, making them ideal for 1D tensor network contraction.

## Decision
We implement a native **Matrix Product State (`MPS`) simulator** (`src/pravaha_q/tensor_network/mps_simulator.py`) representing the $N$-qubit state as a chain of rank-3 tensors $A^{[1]}_{s_1, \alpha_1} A^{[2]}_{\alpha_1, s_2, \alpha_2} \cdots A^{[N]}_{\alpha_{N-1}, s_N}$ with Singular Value Decomposition (SVD) truncation up to maximum bond dimension $\chi_{\max} \in [16, 128]$.

## Consequences
* **Positive:** Allows pre-screening and simulating shallow `BF-DCQO` circuits up to $N = 50+$ qubits on CPU/GPU before dispatching jobs to paid QPU hardware.
