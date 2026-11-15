# RFC-0002: Adaptive Bias-Field Update & Convergence Stopping Protocol (`BF-DCQO`)

* **RFC ID:** RFC-0002
* **Status:** Implemented (`v0.1.0`)
* **Target Module:** `src/pravaha_q/algorithms/bf_dcqo.py`

## 1. Objective
Specify the exact numerical update schedule, learning rate decay, and early-stopping convergence criterion for the iterative Bias-Field Digitized Counterdiabatic Quantum Optimization (`BF-DCQO`) loop.

## 2. Iterative Update Protocol
Given maximum iterations $K_{\max} \in [3, 8]$, initial bias gain $\eta_0 = 0.65$, and inverse temperature sharpness $\gamma_{\text{bias}} = 2.5$:

1. **Initialization ($k = 0$):** Set effective longitudinal field $\tilde{h}_i^{(0)} = h_i^{\text{Ising}}$ for all $i \in \{0, \dots, N-1\}$.
2. **Counterdiabatic Impulse ($k \ge 0$):**
   * Compute analytical AGP coefficient $\alpha_1^{(k)} = -\Gamma_1(\tilde{\mathbf{h}}^{(k)}, \mathbf{J}, \mathbf{K}) / \Gamma_2(\tilde{\mathbf{h}}^{(k)}, \mathbf{J}, \mathbf{K})$.
   * Execute single-step counterdiabatic circuit $\hat{U}_{\text{BF-DCQO}}^{(k)}$ and sample bitstrings + single-qubit magnetizations $m_i^{(k)} = \langle \hat{Z}_i \rangle$.
3. **Sigmoidal Bias Ratchet Update:**
   $$\tilde{h}_i^{(k+1)} = h_i^{\text{Ising}} + \eta_0 \left(1 + \frac{k}{K_{\max}}\right) \tanh\!\left(\gamma_{\text{bias}} \, m_i^{(k)}\right)$$
4. **Polarization Early-Stopping Criterion:**
   Define mean spin polarization $\bar{P}^{(k)} = \frac{1}{N} \sum_{i=1}^{N} \vert{}m_i^{(k)}\vert{}$. If $\bar{P}^{(k)} \ge 0.92$ or the best sampled HUBO energy $E_{\min}^{(k)}$ remains unchanged with $\Delta \bar{P} < 10^{-3}$ for two consecutive steps, terminate early and pass the best bitstring distribution to the QEM + Feasibility Repair pipeline.
