# 🟣 03 — Formal Mathematical Specification
### **Operator Algebra, Proofs, Gauge Potentials & Complexity Bounds**

---

## 1. Theorem 1: Exact Isomorphism Between 3-Local HUBO and Cubic Ising Spin Glass

**Statement:** Let $\mathbf{x} \in \{0, 1\}^N$ be a binary vector and let $E_{\text{HUBO}}(\mathbf{x})$ be any real cubic pseudo-Boolean polynomial:
$$E_{\text{HUBO}}(\mathbf{x}) = c_0 + \sum_{i=1}^{N} L_i x_i + \sum_{1 \le i < j \le N} Q_{ij} x_i x_j + \sum_{1 \le i < j < k \le N} T_{ijk} x_i x_j x_k$$

Under the invertible spectral bijection $\phi: \{0, 1\}^N \to \{+1, -1\}^N$ defined by $z_i = 1 - 2x_i \iff x_i = \frac{1 - z_i}{2}$, the diagonal quantum operator $\hat{H}_C$ acting on $\mathcal{H} = (\mathbb{C}^2)^{\otimes N}$:
$$\hat{H}_C = C_0 \hat{I} + \sum_{i=1}^{N} h_i \hat{Z}_i + \sum_{1 \le i < j \le N} J_{ij} \hat{Z}_i \hat{Z}_j + \sum_{1 \le i < j < k \le N} K_{ijk} \hat{Z}_i \hat{Z}_j \hat{Z}_k$$

satisfies $\langle \mathbf{x} | \hat{H}_C | \mathbf{x} \rangle = E_{\text{HUBO}}(\mathbf{x})$ for all $\mathbf{x} \in \{0, 1\}^N$ if and only if the coefficients satisfy:

$$C_0 = c_0 + \frac{1}{2} \sum_{i=1}^{N} L_i + \frac{1}{4} \sum_{i < j} Q_{ij} + \frac{1}{8} \sum_{i < j < k} T_{ijk}$$

$$h_i = -\frac{1}{2} L_i - \frac{1}{4} \sum_{j \neq i} Q_{\{i,j\}} - \frac{1}{8} \sum_{\substack{j < k \\ j, k \neq i}} T_{\{i,j,k\}}$$

$$J_{ij} = \frac{1}{4} Q_{ij} + \frac{1}{8} \sum_{k \notin \{i,j\}} T_{\{i,j,k\}}, \qquad K_{ijk} = -\frac{1}{8} T_{ijk}$$

**Proof:** Substitute $\hat{X}_{\text{proj}, i} = \frac{1}{2}(\hat{I} - \hat{Z}_i)$ into $E_{\text{HUBO}}$, expand multilinear products using $\hat{Z}_i^2 = \hat{I}$, and collect operators by Pauli weight $w \in \{0, 1, 2, 3\}$. Since $\{ \hat{I}, \hat{Z}_i, \hat{Z}_i \hat{Z}_j, \hat{Z}_i \hat{Z}_j \hat{Z}_k \}$ form an orthogonal basis over the diagonal algebra $\mathcal{D}(\mathbb{C}^{2^N})$ under the Hilbert-Schmidt inner product $\langle A, B \rangle = 2^{-N}\text{Tr}(A^\dagger B)$, the coefficient mapping is unique. $\blacksquare$

---

## 2. Slack-Free Unbalanced Inequality Penalization Bounds

Given an inequality constraint $g(\mathbf{x}) = C_{\max} - \sum_{i \in S} w_i x_i \ge 0$, standard quadratization introduces $B = \lfloor \log_2(C_{\max}) \rfloor + 1$ binary slack variables $s_b \in \{0, 1\}$:
$$\mathcal{P}_{\text{slack}}(\mathbf{x}, \mathbf{s}) = \lambda \left( g(\mathbf{x}) - \sum_{b=0}^{B-1} 2^b s_b \right)^2$$

In `PravahaQ`, we eliminate $\mathbf{s}$ completely via the **Unbalanced Penalization Functional** (Montañez-Barrera et al., 2024):
$$\Phi_{\text{unbal}}(g(\mathbf{x})) \approx \lambda_0 \left( e^{-\beta g(\mathbf{x})} - 1 \right) \approx -\lambda_1 g(\mathbf{x}) + \lambda_2 g(\mathbf{x})^2$$

### Stationarity & Feasibility Condition
The local minimum of $\Phi_{\text{unbal}}(g)$ occurs at $g^* = \frac{\lambda_1}{2\lambda_2} > 0$. By selecting $\lambda_1, \lambda_2$ such that $\frac{\lambda_1}{2\lambda_2} = \bar{g}_{\text{target}}$ (expected thermal headroom) and $\lambda_2 > \frac{\Delta E_{\max}}{\delta_{\min}^2}$ where $\delta_{\min}$ is the smallest infeasible violation step, any state violating $g(\mathbf{x}) < 0$ incurs a strictly positive energy penalty $\Delta \Phi > \Delta E_{\max}$ without adding a single ancilla qubit.

---

## 3. Variational Adiabatic Gauge Potential (AGP) for `BF-DCQO`

Let the instantaneous adiabatic Hamiltonian with longitudinal bias field $\mathbf{h}^{(k)}$ be:
$$\hat{H}_{\text{ad}}(\lambda) = (1 - \lambda) \underbrace{\left( -\sum_{i=1}^{N} \hat{X}_i + \sum_{i=1}^{N} h_i^{\text{bias},(k)} \hat{Z}_i \right)}_{\hat{H}_0^{(k)}} + \lambda \hat{H}_C$$

The exact Adiabatic Gauge Potential $\hat{\mathcal{A}}_\lambda$ satisfies the operator partial differential equation:
$$\left[ \hat{H}_{\text{ad}}(\lambda), \, \partial_\lambda \hat{H}_{\text{ad}}(\lambda) - i \big[\hat{H}_{\text{ad}}(\lambda), \hat{\mathcal{A}}_\lambda\big] \right] = 0$$

Using the **1st-order Claeys–Berry–Sels Nested Commutator Ansatz** $\hat{\mathcal{A}}_\lambda^{(1)} = i \alpha_1(\lambda) [\hat{H}_{\text{ad}}, \partial_\lambda \hat{H}_{\text{ad}}]$, and evaluating the Pauli commutators $[\hat{X}_i, \hat{Z}_i] = -2i\hat{Y}_i$, we obtain:

$$\hat{\mathcal{A}}_\lambda^{(1)} = 2\alpha_1(\lambda) \left( \sum_{i=1}^{N} \tilde{h}_i \hat{Y}_i + \sum_{i < j} J_{ij} (\hat{Y}_i \hat{Z}_j + \hat{Z}_i \hat{Y}_j) + \sum_{i < j < k} K_{ijk} (\hat{Y}_i \hat{Z}_j \hat{Z}_k + \hat{Z}_i \hat{Y}_j \hat{Z}_k + \hat{Z}_i \hat{Z}_j \hat{Y}_k) \right)$$

### Closed-Form Optimal Gauge Coefficient $\alpha_1^*(\lambda)$
Minimizing the Hilbert-Schmidt action norm $\mathcal{S}(\alpha_1) = 2^{-N} \text{Tr}[\hat{G}_\lambda(\alpha_1)^2]$ yields the exact analytical expression (zero classical optimization required!):

$$\alpha_1^*(\lambda) = -\frac{\Gamma_1(\lambda)}{\Gamma_2(\lambda)}$$

where:
$$\Gamma_1(\lambda) = \sum_{i=1}^{N} \tilde{h}_i^2 + 2 \sum_{i < j} J_{ij}^2 + 3 \sum_{i < j < k} K_{ijk}^2$$
$$\Gamma_2(\lambda) = 4(1-\lambda)^2 \left( \sum_i \tilde{h}_i^2 + 4\sum_{i<j} J_{ij}^2 + 9\sum_{i<j<k} K_{ijk}^2 \right) + 4\lambda^2 \left( \sum_i \tilde{h}_i^4 + 6\sum_{i<j} J_{ij}^4 + \mathcal{O}(J^2 h^2) \right)$$

In the fast impulse regime ($\dot{\lambda} \gg 1$), the evolution unitary reduces to the counterdiabatic impulse $\hat{U}_{\text{CD}}(\theta_{\text{CD}}) = \exp(-i \theta_{\text{CD}} \hat{\mathcal{A}}_\lambda^{(1)})$, followed by the longitudinal bias update:
$$h_i^{\text{bias},(k+1)} = h_i^{\text{bias},(k)} + \eta_k \tanh\!\left(\gamma \langle \psi^{(k)} | \hat{Z}_i | \psi^{(k)} \rangle\right)$$

---

## 4. Quantum Error Mitigation Mathematics (`ZNE` + `TREX` + `CVaR`)

1. **Conditional Value-at-Risk ($\text{CVaR}_\alpha$):** Given $S$ measured bitstrings ordered by non-decreasing energy $E(z_{(1)}) \le E(z_{(2)}) \le \dots \le E(z_{(S)})$ and tail fraction $\alpha \in (0, 1]$:
   $$\text{CVaR}_\alpha(E) = \frac{1}{\lceil \alpha S \rceil} \sum_{s=1}^{\lceil \alpha S \rceil} E(z_{(s)})$$
2. **Zero-Noise Extrapolation (Richardson Extrapolation):** Executing unitary folded circuits $\hat{U}_k = \hat{U} (\hat{U}^\dagger \hat{U})^{(c_k - 1)/2}$ at noise scale factors $\{c_1, c_2, \dots, c_M\}$ (e.g., $c \in \{1, 3, 5\}$), the zero-noise estimator is:
   $$\hat{E}_{\text{ZNE}} = \sum_{m=1}^{M} \gamma_m \langle \hat{H}_C \rangle_{c_m}, \qquad \text{where } \gamma_m = \prod_{j \neq m} \frac{c_j}{c_j - c_m}$$
