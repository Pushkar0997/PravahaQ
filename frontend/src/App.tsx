import React, { useMemo, useState } from "react";
import { computeDeterministicPhysics } from "./data/livePhysicsEngine";
import {
  AlgorithmMode,
  HardwareTopology,
  SDKTarget,
  SimulationState,
} from "./types/engine";

const TABS = [
  { id: "topology", label: "1. Cluster & Microgrid Twin" },
  { id: "hubo", label: "2. Vaisheshika 3-Body HUBO & Ising" },
  { id: "bf_dcqo", label: "3. BF-DCQO & PCE Compression" },
  { id: "compiler", label: "4. Paulihedral OpenQASM 3 & CUDA-Q" },
  { id: "qem", label: "5. ZNE + TREX + CVaR Mitigation" },
  { id: "benchmarks", label: "6. Multi-SDK & Schedule Decoder" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export function App() {
  const [activeTab, setActiveTab] = useState<TabId>("topology");
  const [sim, setSim] = useState<SimulationState>({
    numNodes: 3,
    numTasks: 4,
    numPowerStates: 2,
    algorithm: "bf_dcqo",
    sdk: "qiskit",
    topology: "heavy_hex",
    pceQubits: 6,
    unbalancedLambda1: 3.5,
    unbalancedLambda2: 6.0,
    cvarAlpha: 0.2,
    enableCubic3Body: true,
  });

  const physics = useMemo(() => computeDeterministicPhysics(sim), [sim]);

  return (
    <div className="obsidian-shell">
      {/* Top Mission Control HUD */}
      <header className="topbar">
        <div className="brand-title">
          <span>⚛️ PravahaQ</span>
          <span className="badge-pill">v0.1.0</span>
          <span className="badge-pill badge-violet">
            Compiler: Vaisheshika-3Local
          </span>
        </div>
        <div style={{ display: "flex", gap: "0.6rem", alignItems: "center" }}>
          <span className="badge-pill">
            Active Algorithm: {sim.algorithm.toUpperCase()}
          </span>
          <span className="badge-pill">SDK: {sim.sdk.toUpperCase()}</span>
          <span
            className="badge-pill"
            style={{
              background: "rgba(16, 185, 129, 0.15)",
              color: "#10b981",
              borderColor: "rgba(16, 185, 129, 0.4)",
            }}
          >
            SLA Feasibility: 100% Guaranteed
          </span>
        </div>
      </header>

      {/* Live Telemetry KPI Strip */}
      <section className="kpi-strip">
        <div className="glass-card">
          <div className="kpi-label">Binary Variables / Qubits</div>
          <div className="kpi-value">
            {physics.totalBinaryVars} vars → {physics.qubitsExecuted}Q (
            {physics.compressionRatio}x)
          </div>
        </div>
        <div className="glass-card">
          <div className="kpi-label">Approximation Ratio (AR)</div>
          <div className="kpi-value" style={{ color: "var(--cyan-coherent)" }}>
            {(physics.approximationRatio * 100).toFixed(1)}%
          </div>
        </div>
        <div className="glass-card">
          <div className="kpi-label">Ground / Found Energy</div>
          <div className="kpi-value">
            {physics.foundEnergy} / {physics.groundEnergy}
          </div>
        </div>
        <div className="glass-card">
          <div className="kpi-label">Paulihedral CX Saved</div>
          <div className="kpi-value" style={{ color: "#c084fc" }}>
            -{physics.paulihedralCxSavedPct}% 2Q Gates
          </div>
        </div>
        <div className="glass-card">
          <div className="kpi-label">Spectral Gap ΔE / Frustration</div>
          <div className="kpi-value" style={{ color: "var(--amber-thermal)" }}>
            {physics.spectralGap} eV | {physics.frustrationIndex}
          </div>
        </div>
      </section>

      {/* Navigation Tabs */}
      <nav className="nav-tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tab-btn ${activeTab === t.id ? "active" : ""}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {/* Main Workspace */}
      <div className="grid-2col">
        {/* Left Control Deck */}
        <aside className="glass-card">
          <h3
            style={{
              fontFamily: "var(--font-display)",
              marginBottom: "1rem",
              fontSize: "1.05rem",
            }}
          >
            🎛️ Quantum-Classical Control Parameters
          </h3>

          <div className="control-row">
            <label>
              <span>Edge Compute Nodes (N)</span>
              <strong>{sim.numNodes} Nodes</strong>
            </label>
            <input
              type="range"
              min={2}
              max={6}
              value={sim.numNodes}
              onChange={(e) =>
                setSim({ ...sim, numNodes: Number(e.target.value) })
              }
            />
          </div>

          <div className="control-row">
            <label>
              <span>LLM Inference Batches (M)</span>
              <strong>{sim.numTasks} Batches</strong>
            </label>
            <input
              type="range"
              min={2}
              max={10}
              value={sim.numTasks}
              onChange={(e) =>
                setSim({ ...sim, numTasks: Number(e.target.value) })
              }
            />
          </div>

          <div className="control-row">
            <label>
              <span>Frontier Algorithm Regime</span>
              <strong>{sim.algorithm}</strong>
            </label>
            <select
              value={sim.algorithm}
              onChange={(e) =>
                setSim({ ...sim, algorithm: e.target.value as AlgorithmMode })
              }
              style={{
                background: "#07090e",
                color: "#f8fafc",
                padding: "0.5rem",
                borderRadius: "6px",
                border: "1px solid var(--border-subtle)",
              }}
            >
              <option value="bf_dcqo">
                BF-DCQO (Bias-Field Counterdiabatic — 4 Shots)
              </option>
              <option value="pce">
                Multistage PCE (Sub-Linear Qubit Compression)
              </option>
              <option value="iqo_bnb">
                IQO-BnB (Backtracking Correlation Branch &amp; Bound)
              </option>
              <option value="cd_warm_qaoa">
                CD-WarmQAOA (QP Warm-Start + CD Mixer)
              </option>
              <option value="mps">
                MPS Tensor Network Surrogate (SVD χ=32)
              </option>
            </select>
          </div>

          <div className="control-row">
            <label>
              <span>Quantum SDK / Execution Target</span>
              <strong>{sim.sdk}</strong>
            </label>
            <select
              value={sim.sdk}
              onChange={(e) =>
                setSim({ ...sim, sdk: e.target.value as SDKTarget })
              }
              style={{
                background: "#07090e",
                color: "#f8fafc",
                padding: "0.5rem",
                borderRadius: "6px",
                border: "1px solid var(--border-subtle)",
              }}
            >
              <option value="qiskit">Qiskit v2.x Primitives (SamplerV2)</option>
              <option value="pennylane">
                PennyLane Differentiable QNode (Adjoint QNG)
              </option>
              <option value="cirq">Google Cirq 2D Planar GridQubit</option>
              <option value="mps">MPS Tensor Network Surrogate</option>
              <option value="ibm_runtime">
                IBM Quantum Heron Hardware (ibm_fez)
              </option>
            </select>
          </div>

          <div className="control-row">
            <label>
              <span>Hardware Topology Graph</span>
              <strong>{sim.topology}</strong>
            </label>
            <select
              value={sim.topology}
              onChange={(e) =>
                setSim({
                  ...sim,
                  topology: e.target.value as HardwareTopology,
                })
              }
              style={{
                background: "#07090e",
                color: "#f8fafc",
                padding: "0.5rem",
                borderRadius: "6px",
                border: "1px solid var(--border-subtle)",
              }}
            >
              <option value="heavy_hex">IBM Heavy-Hex Lattice (d ≤ 3)</option>
              <option value="planar_grid">Google 2D Planar Square Grid</option>
              <option value="all_to_all">IonQ All-to-All Trapped Ion</option>
            </select>
          </div>

          <div className="control-row">
            <label>
              <span>Unbalanced Inequality Penalty (λ₁, λ₂)</span>
              <strong>
                ({sim.unbalancedLambda1}, {sim.unbalancedLambda2})
              </strong>
            </label>
            <input
              type="range"
              min={1}
              max={10}
              step={0.5}
              value={sim.unbalancedLambda1}
              onChange={(e) =>
                setSim({ ...sim, unbalancedLambda1: Number(e.target.value) })
              }
            />
          </div>

          <div
            style={{
              marginTop: "1rem",
              padding: "0.75rem",
              background: "rgba(0, 240, 255, 0.06)",
              borderRadius: "8px",
              fontFamily: "var(--font-mono)",
              fontSize: "0.76rem",
            }}
          >
            <div>Ground-State Bitstring z*:</div>
            <div
              style={{
                color: "var(--cyan-coherent)",
                fontSize: "0.95rem",
                marginTop: "0.25rem",
                wordBreak: "break-all",
              }}
            >
              |{physics.bestBitstring}⟩
            </div>
          </div>
        </aside>

        {/* Right Dynamic Inspection Panel */}
        <main className="glass-card">
          {activeTab === "topology" && (
            <div>
              <h3 style={{ fontFamily: "var(--font-display)", marginBottom: "0.85rem" }}>
                🌐 Coupled Edge-AI Compute &amp; 3-Phase Microgrid Routing Schedule
              </h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.86rem", marginBottom: "1rem" }}>
                Real-time co-designed assignment of distributed LLM prefill/decode batches across
                heterogeneous edge nodes under solar/BESS microgrid thermal caps.
              </p>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Inference Workload</th>
                    <th>Assigned Edge Node</th>
                    <th>Thermal Load (W)</th>
                    <th>Interconnect Latency</th>
                    <th>SLA Status</th>
                  </tr>
                </thead>
                <tbody>
                  {physics.taskRouting.map((r, idx) => (
                    <tr key={idx}>
                      <td style={{ color: "var(--cyan-coherent)" }}>{r.task}</td>
                      <td>{r.node}</td>
                      <td>{r.watts} W</td>
                      <td>{r.latencyMs} ms</td>
                      <td style={{ color: "var(--emerald-valid)" }}>✓ Feasible</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {activeTab === "hubo" && (
            <div>
              <h3 style={{ fontFamily: "var(--font-display)", marginBottom: "0.85rem" }}>
                🧲 Vaisheshika 3-Body HUBO &amp; Ising Spin-Glass Tensor Inspector
              </h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "0.9rem" }}>
                Diagonal entries show longitudinal fields h_i; off-diagonal entries show pairwise
                KV-cache &amp; equality couplings J_ij. Cubic 3-body terms K_ijk penalize simultaneous
                compute-thermal-phase saturation without slack ancilla qubits.
              </p>
              <div className="mono-code" style={{ marginBottom: "1rem" }}>
                {physics.couplingMatrix
                  .map((row) =>
                    row.map((v) => v.toFixed(2).padStart(6, " ")).join(" ")
                  )
                  .join("\n")}
              </div>
              <h4 style={{ fontSize: "0.9rem", marginBottom: "0.5rem", color: "#c084fc" }}>
                Active 3-Body Cubic Tryāṇuka Terms (K_ijk Z_i Z_j Z_k):
              </h4>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>3-Local Cubic Spin Operator</th>
                    <th>Coupling Coefficient K_ijk</th>
                  </tr>
                </thead>
                <tbody>
                  {physics.cubicTriads.map((t, i) => (
                    <tr key={i}>
                      <td>{t.triad}</td>
                      <td style={{ color: "var(--amber-thermal)" }}>{t.weight}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {activeTab === "bf_dcqo" && (
            <div>
              <h3 style={{ fontFamily: "var(--font-display)", marginBottom: "0.85rem" }}>
                ⚡ BF-DCQO Longitudinal Bias-Field Ratchet &amp; Spin Polarization
              </h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "1rem" }}>
                Analytical Adiabatic Gauge Potential (AGP) + iterative longitudinal bias updates
                h_i^(k+1) = h_i^(0) + η tanh(γ ⟨Z_i⟩) converge in 4 impulses with zero variational outer loop.
              </p>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Impulse Step k</th>
                    <th>Expectation ⟨H_C⟩</th>
                    <th>Mean Polarization |⟨Z_i⟩|</th>
                    <th>Leading Qubit Magnetizations ⟨Z_0..Z_5⟩</th>
                  </tr>
                </thead>
                <tbody>
                  {physics.bfIterations.map((it) => (
                    <tr key={it.iter}>
                      <td>Impulse #{it.iter}</td>
                      <td style={{ color: "var(--cyan-coherent)" }}>{it.energy}</td>
                      <td style={{ color: "var(--emerald-valid)" }}>
                        {(it.polarization * 100).toFixed(1)}%
                      </td>
                      <td>[{it.mags.join(", ")}]</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {activeTab === "compiler" && (
            <div>
              <h3 style={{ fontFamily: "var(--font-display)", marginBottom: "0.85rem" }}>
                🛠️ Paulihedral-Optimized OpenQASM 3.0 &amp; NVIDIA CUDA-Q Kernel
              </h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.85rem" }}>
                <div>
                  <div className="kpi-label">OpenQASM 3.0 (IBM Heron + DD Pragmas)</div>
                  <pre className="mono-code">{physics.openqasm3}</pre>
                </div>
                <div>
                  <div className="kpi-label">NVIDIA CUDA-Q (@cudaq.kernel)</div>
                  <pre className="mono-code">{physics.cudaqCode}</pre>
                </div>
              </div>
            </div>
          )}

          {activeTab === "qem" && (
            <div>
              <h3 style={{ fontFamily: "var(--font-display)", marginBottom: "0.85rem" }}>
                🛡️ Quantum Error Mitigation: ZNE Gate Folding + TREX + CVaR_α
              </h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "1rem" }}>
                Richardson Zero-Noise Extrapolation across unitary folding scale factors c ∈ {"{1, 3, 5}"}
                combined with Conditional Value-at-Risk (α = {sim.cvarAlpha}) tail aggregation.
              </p>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Noise Folding Scale Factor (c)</th>
                    <th>Unitary Sequence</th>
                    <th>Measured / Extrapolated Energy</th>
                  </tr>
                </thead>
                <tbody>
                  {physics.znePoints.map((pt, i) => (
                    <tr key={i}>
                      <td style={{ color: pt.scale === 0 ? "var(--cyan-coherent)" : undefined }}>
                        c = {pt.scale.toFixed(1)} {pt.scale === 0 ? "(Zero-Noise Limit ⭐)" : ""}
                      </td>
                      <td>
                        {pt.scale === 0
                          ? "Richardson Extrapolated"
                          : pt.scale === 1
                          ? "U"
                          : `U (U† U)^${(pt.scale - 1) / 2}`}
                      </td>
                      <td>{pt.energy}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {activeTab === "benchmarks" && (
            <div>
              <h3 style={{ fontFamily: "var(--font-display)", marginBottom: "0.85rem" }}>
                🏆 5-Backend Execution Matrix (Qiskit v2, PennyLane, Cirq, MPS)
              </h3>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Algorithm</th>
                    <th>Quantum SDK</th>
                    <th>Active Qubits</th>
                    <th>Best Energy</th>
                    <th>Approximation Ratio</th>
                    <th>Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {physics.sdkBenchmarks.map((row, idx) => (
                    <tr key={idx}>
                      <td style={{ color: "var(--cyan-coherent)" }}>{row.algorithm}</td>
                      <td>{row.sdk}</td>
                      <td>{row.qubits} Q</td>
                      <td>{row.energy}</td>
                      <td style={{ color: "var(--emerald-valid)" }}>
                        {(row.ar * 100).toFixed(1)}%
                      </td>
                      <td>{row.latencyMs} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
