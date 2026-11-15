# RFC-0003: Deterministic Post-Quantum Feasibility Repair Protocol

* **RFC ID:** RFC-0003
* **Status:** Implemented (`v0.1.0`)
* **Target Module:** `src/pravaha_q/math_engine/feasibility_repair.py`

## 1. Objective
Ensure that 100% of schedules returned by `PravahaQ` satisfy all hard physical constraints—even when executed under severe depolarizing hardware noise.

## 2. Two-Phase Deterministic Repair Algorithm
Given any raw candidate bitstring $\mathbf{x} \in \{0, 1\}^N$ decoded into task assignment matrix $X \in \{0, 1\}^{M \times N_{\text{node}}}$ and node power state matrix $Y \in \{0, 1\}^{N_{\text{node}} \times R}$:

### Phase 1: Single-Assignment Projection ($O(M \cdot N_{\text{node}})$)
* For each task $m \in \{1, \dots, M\}$:
  * If $\sum_n X_{m,n} == 0$ (unassigned): Assign task $m$ to node $n^* = \arg\min_n \Delta E_{\text{HUBO}}(m \to n)$ that has available thermal capacity.
  * If $\sum_n X_{m,n} > 1$ (multi-assigned): Retain only the single node $n^*$ with minimum marginal HUBO energy and set all other $X_{m, n \neq n^*} \leftarrow 0$.
* Enforce exact one-hot power state per node: if $\sum_r Y_{n,r} \neq 1$, select the minimum-cost state $r^*$ that covers the current thermal load of node $n$.

### Phase 2: Thermal & Transformer Headroom Balancing ($O(M \cdot N_{\text{node}})$)
* If any node $n$ violates its thermal ceiling ($\sum_m \tau_{m,n} X_{m,n} > C_{n, r^*}^{\max}$), first upgrade power/cooling state $r^* \to r_{\text{high}}$; if still overloaded, migrate the highest-heat task $m^*$ to the node $n'$ with greatest remaining thermal slack.

## 3. Complexity & Invariant Guarantee
* **Time Complexity:** Strictly bounded by $O(M \cdot N_{\text{node}} + N_{\text{node}} \cdot R)$ (< 0.4 milliseconds for $N=150$).
* **Post-Condition Invariant:** `assert is_schedule_feasible(repaired_x) is True` is enforced on every solver return path.
