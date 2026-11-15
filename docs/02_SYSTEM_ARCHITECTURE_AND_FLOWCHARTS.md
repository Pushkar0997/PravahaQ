# 🔵 02 — System Architecture & Execution Flowcharts
### **End-to-End C4 Architecture, Compiler Dataflow & Quantum Control Loops**

---

## 1. High-Level Hexagonal Architecture (Ports & Adapters)

PravahaQ enforces strict **Hexagonal Architecture (Clean Architecture)**. The core mathematical domain (`domain/`, `math_engine/`, `algorithms/`) has **zero hard dependencies** on external quantum SDKs or web frameworks. All SDKs (`Qiskit`, `PennyLane`, `Cirq`, `MPS`, `CUDA-Q`) plug in as interchangeable outer adapters.

```mermaid
flowchart TB
    subgraph Clients["External Clients & Interfaces"]
        UI["🎨 Quantum Obsidian React UI<br/>(8-Panel Mission Control)"]
        CLI["💻 pravaha-q CLI<br/>(Terminal Automation)"]
        SDK["🐍 Python Developer SDK<br/>(PravahaClient)"]
        NB["📓 12 Jupyter Notebooks<br/>(Research & Mini-Projects)"]
    end

    subgraph ControlPlane["FastAPI Control Plane & Infrastructure"]
        API["⚡ FastAPI REST & WebSocket Router"]
        FSM["🔄 Async Job Finite State Machine"]
        CACHE["🗄️ SHA-256 Content-Addressable Circuit Cache"]
    end

    subgraph CoreEngine["PravahaQ Pure Mathematical & Algorithmic Core"]
        DOM["Cluster & 3-Phase Microgrid Digital Twin"]
        HUBO["Vaisheshika 3-Body HUBO & Unbalanced Penalty Compiler"]
        ALG["Algorithm Engine:<br/>BF-DCQO | Multistage PCE | IQO-BnB | CD-WarmQAOA"]
        COMP["Paulihedral Compiler & Topology SWAP Router"]
        QEM["Quantum Error Mitigation:<br/>ZNE Gate Folding + TREX Readout + CVaR_α"]
        REP["🛡️ Deterministic Post-Quantum Feasibility Repair"]
    end

    subgraph Adapters["5-Backend Execution Matrix (Adapters)"]
        QK["Qiskit v2.x Adapter<br/>(SamplerV2 / IBM Heron Runtime)"]
        PL["PennyLane Adapter<br/>(Adjoint QNG Differentiable QNode)"]
        CQ["Google Cirq Adapter<br/>(2D GridQubit Simulator)"]
        MPS["MPS Tensor Network<br/>(SVD Classical Surrogate)"]
        QIR["OpenQASM 3.0 &<br/>NVIDIA CUDA-Q Emitter"]
    end

    UI & CLI & SDK & NB --> API
    API --> FSM & CACHE
    FSM --> DOM --> HUBO --> ALG --> COMP
    COMP --> QK & PL & CQ & MPS & QIR
    QK & PL & CQ & MPS --> QEM --> REP
```

---

## 2. Flowchart 2: The *Vaisheshika* 3-Body HUBO & Unbalanced Penalty Compiler

```mermaid
flowchart LR
    A["Workload + Grid Trace"] --> B["Construct Binary Vector x ∈ {0,1}^N"]
    B --> C["Build 1-Body Linear Vector L_i<br/>(Base Latency + Tariff)"]
    B --> D["Build 2-Body Matrix Q_ij<br/>(KV-Cache Sync + Equality Penalties)"]
    B --> E["Build 3-Body Cubic Tensor T_ijk<br/>(Compute-Thermal-Phase Overload)"]
    B --> F["Encode Inequality Constraints g_n(x) ≥ 0<br/>via Unbalanced Penalty: -λ_1 g_n + λ_2 g_n^2"]
    C & D & E & F --> G["Aggregate Total HUBO Polynomial"]
    G --> H["Apply Spectral Projection x_i = (1 - Z_i)/2"]
    H --> I["Normalize Spectral Norm ||H_C||_∞ ≤ 1<br/>Output Ising Tensors (C_0, h_i, J_ij, K_ijk)"]
```

---

## 3. Flowchart 3: Multistage Pauli Correlation Encoding (`PCE`) Pipeline

When the number of binary variables $N$ exceeds the available physical qubits $n$ (e.g., $N = 84$ variables on $n = 7$ qubits), PravahaQ automatically routes execution through the **Multistage PCE Pipeline**:

```mermaid
flowchart TD
    A["Input: N Binary Variables<br/>Target Hardware: n << N Qubits"] --> B["Generate Commuting/Non-Commuting<br/>k-Body Pauli Strings {Π_1, ..., Π_N}<br/>where n ≥ O(N^{1/k})"]
    B --> C["Initialize Brickwall Hardware-Efficient<br/>Ansatz |ψ(θ)⟩ with Depth L"]
    C --> D["Stage s = 1..S Curriculum Loop:<br/>Set Slope α_s = 2^s, Regularizer β_s"]
    D --> E["Measure Pauli Correlators:<br/>c_m(θ) = ⟨ψ(θ)| Π_m |ψ(θ)⟩"]
    E --> F["Compute Smooth Spin Relaxation:<br/>z̃_m(θ) = tanh(α_s · c_m(θ))"]
    F --> G["Evaluate Differentiable Loss:<br/>L_PCE = E_Ising(z̃) + β_s ∑(1 - z̃_m^2)"]
    G --> H{"Stage Converged?"}
    H -- "No" --> I["Update θ via Adjoint Gradient / QNG"] --> E
    H -- "Yes, Next Stage s+1" --> D
    H -- "Final Stage Complete" --> J["Project Hard Signs: z_m* = sgn(⟨Π_m⟩)<br/>Run Deterministic Feasibility Repair"]
```

---

## 4. Flowchart 4: Iterative Quantum Branch-and-Bound (`IQO-BnB`) + Feasibility Shield

```mermaid
flowchart TD
    A["Full N-Variable Spin-Glass Graph G_0"] --> B["Execute Shallow BF-DCQO Circuit<br/>on Current Graph G_t"]
    B --> C["Compute 2-Point Correlation Matrix:<br/>M_ij = ⟨Z_i Z_j⟩ - ⟨Z_i⟩⟨Z_j⟩"]
    C --> D["Rank Top-B Strongest Spin Pairs<br/>|(i*, j*)| with Largest |M_ij|"]
    D --> E["Branch 1 (Primary):<br/>Freeze Z_j = sgn(M_ij) · Z_i"]
    D --> F["Branch 2 (Backtrack Queue):<br/>Freeze Opposite Parity Z_j = -sgn(M_ij) · Z_i"]
    E & F --> G["Substitute & Eliminate Variable j:<br/>Reduce Graph Size N -> N - 1"]
    G --> H{"Remaining Variables<br/>N ≤ N_cutoff?"}
    H -- "No" --> B
    H -- "Yes" --> I["Solve Reduced Core Exactly<br/>& Prune Suboptimal Branches"]
    I --> J["🛡️ Post-Quantum Feasibility Repair:<br/>Fix Any Violated One-Hot or Thermal Caps"]
    J --> K["Return 100% SLA-Valid Schedule"]
```
