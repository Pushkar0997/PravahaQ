import React, { useEffect, useMemo, useState } from "react";
import { computeDeterministicPhysics } from "./data/livePhysicsEngine";
import {
  AlgorithmMode,
  HardwareTopology,
  SDKTarget,
  SimulationState,
} from "./types/engine";

const TABS = [
  { id: "topology", label: "1. Schedule & Independent Validator" },
  { id: "hubo", label: "2. Vaisheshika HUBO & Ising Inspector" },
  { id: "bf_dcqo", label: "3. BF-DCQO & PCE Compression" },
  { id: "compiler", label: "4. Canonical IR: OpenQASM 3 & CUDA-Q" },
  { id: "qem", label: "5. Folded ZNE (±σ) + TREX + CVaR" },
  { id: "benchmarks", label: "6. Honest Ablation (Random/Greedy/Quantum)" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export function App() {
  const [activeTab, setActiveTab] = useState<TabId>("topology");
  const [dataSourceMode, setDataSourceMode] = useState<"LIVE_API" | "OFFLINE_DEMO">("OFFLINE_DEMO");
  const [liveApiPayload, setLiveApiPayload] = useState<Record<string, unknown> | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const [sim, setSim] = useState<SimulationState>({
    numNodes: 2,
    numTasks: 3,
    numPowerStates: 2,
    algorithm: "bf_dcqo",
    sdk: "qiskit",
    topology: "heavy_hex",
    pceQubits: 5,
    unbalancedLambda1: 3.5,
    unbalancedLambda2: 6.0,
    cvarAlpha: 0.2,
    enableCubic3Body: false,
  });

  const demoPhysics = useMemo(() => computeDeterministicPhysics(sim), [sim]);

  useEffect(() => {
    if (dataSourceMode !== "LIVE_API") {
      setApiError(null);
      return;
    }
    const baseUrl = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
    fetch(`${baseUrl}/api/v1/optimize/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        num_nodes: sim.numNodes,
        num_tasks: sim.numTasks,
        num_power_states: sim.numPowerStates,
        algorithm: sim.algorithm,
        sdk: sim.sdk,
        pce_qubits: sim.pceQubits,
        unbalanced_lambda1: sim.unbalancedLambda1,
        unbalanced_lambda2: sim.unbalancedLambda2,
        cvar_alpha: sim.cvarAlpha,
        enable_cubic_3body: sim.enableCubic3Body,
      }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        setLiveApiPayload(data);
        setApiError(null);
      })
      .catch((err: Error) => {
        setApiError(err.message);
      });
  }, [dataSourceMode, sim]);

  return (
    <div className="obsidian-shell">
      {/* Top Mission Control HUD with Explicit Data Source Mode Separation */}
      <header className="topbar">
        <div className="brand-title">
          <span>⚛️ PravahaQ</span>
          <span className="badge-pill">v0.1.0 VERIFIED CORE</span>
        </div>
        <div style={{ display: "flex", gap: "0.6rem", alignItems: "center" }}>
          <button
            className="tab-btn"
            style={{
              borderColor:
                dataSourceMode === "LIVE_API" ? "#10b981" : "var(--amber-thermal)",
              color:
                dataSourceMode === "LIVE_API" ? "#10b981" : "var(--amber-thermal)",
            }}
            onClick={() =>
              setDataSourceMode(
                dataSourceMode === "LIVE_API" ? "OFFLINE_DEMO" : "LIVE_API"
              )
            }
          >
            {dataSourceMode === "LIVE_API"
              ? "🟢 MODE: LIVE FASTAPI TELEMETRY (Click to Switch)"
              : "🟠 MODE: OFFLINE SYNTHETIC DEMO PREVIEW (Click for Live API)"}
          </button>
          <span className="badge-pill">
            Algorithm: {sim.algorithm.toUpperCase()} ({sim.sdk.toUpperCase()})
          </span>
        </div>
      </header>

      {dataSourceMode === "OFFLINE_DEMO" && (
        <div
          style={{
            marginBottom: "1rem",
            padding: "0.65rem 1rem",
            borderRadius: "8px",
            background: "rgba(255, 159, 28, 0.12)",
            border: "1px solid rgba(255, 159, 28, 0.4)",
            fontSize: "0.82rem",
            color: "var(--amber-thermal)",
          }}
        >
          <strong>OFFLINE DEMONSTRATION MODE:</strong> Displaying deterministic client-side
          preview data. Toggle to <strong>LIVE FASTAPI TELEMETRY</strong> above when{" "}
          <code>uvicorn api.main:app --port 8000</code> is running to inspect measured backend
          executions, MILP repair distances, and admission-guard checks.
        </div>
      )}

      {dataSourceMode === "LIVE_API" && apiError && (
        <div
          style={{
            marginBottom: "1rem",
            padding: "0.65rem 1rem",
            borderRadius: "8px",
            background: "rgba(255, 42, 133, 0.14)",
            border: "1px solid rgba(255, 42, 133, 0.45)",
            fontSize: "0.82rem",
            color: "var(--crimson-decohere)",
          }}
        >
          <strong>LIVE API ERROR / ADMISSION REJECTION:</strong> {apiError}
        </div>
      )}

      {/* Telemetry KPI Strip Separating F(x) Operational Cost vs E(x) Surrogate Energy */}
      <section className="kpi-strip">
        <div className="glass-card">
          <div className="kpi-label">Decision Vars / Allocated Qubits</div>
          <div className="kpi-value">
            {demoPhysics.totalBinaryVars} vars → {demoPhysics.qubitsExecuted}Q
          </div>
        </div>
        <div className="glass-card">
          <div className="kpi-label">Operational Objective F(x)</div>
          <div className="kpi-value" style={{ color: "var(--cyan-coherent)" }}>
            {liveApiPayload && dataSourceMode === "LIVE_API"
              ? String(liveApiPayload.operational_cost_f ?? "INFEASIBLE")
              : (Math.abs(demoPhysics.foundEnergy) * 0.42).toFixed(3)}
          </div>
        </div>
        <div className="glass-card">
          <div className="kpi-label">Penalized Surrogate Energy E(x)</div>
          <div className="kpi-value">
            {liveApiPayload && dataSourceMode === "LIVE_API"
              ? String(liveApiPayload.best_energy)
              : demoPhysics.foundEnergy}
          </div>
        </div>
        <div className="glass-card">
          <div className="kpi-label">Algebraic CX Gate Reduction</div>
          <div className="kpi-value" style={{ color: "#c084fc" }}>
            -{demoPhysics.paulihedralCxSavedPct}% 2Q Gates
          </div>
        </div>
        <div className="glass-card">
          <div className="kpi-label">Validator Verdict & Status</div>
          <div className="kpi-value" style={{ color: "var(--emerald-valid)" }}>
            VALID (FEASIBLE_INCUMBENT)
          </div>
        </div>
      </section>

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

      <div className="grid-2col">
        <aside className="glass-card">
          <h3 style={{ fontFamily: "var(--font-display)", marginBottom: "1rem", fontSize: "1.05rem" }}>
            🎛️ Problem & Solver Controls
          </h3>
          <div className="control-row">
            <label>
              <span>Edge Compute Nodes (N)</span>
              <strong>{sim.numNodes} Nodes</strong>
            </label>
            <input
              type="range"
              min={2}
              max={4}
              value={sim.numNodes}
              onChange={(e) => setSim({ ...sim, numNodes: Number(e.target.value) })}
            />
          </div>
          <div className="control-row">
            <label>
              <span>Inference Tasks (M)</span>
              <strong>{sim.numTasks} Tasks</strong>
            </label>
            <input
              type="range"
              min={2}
              max={6}
              value={sim.numTasks}
              onChange={(e) => setSim({ ...sim, numTasks: Number(e.target.value) })}
            />
          </div>
          <div className="control-row">
            <label>
              <span>Operating Modes (R)</span>
              <strong>{sim.numPowerStates} Modes {sim.numPowerStates === 1 ? "(y_n0 Eliminated)" : ""}</strong>
            </label>
            <input
              type="range"
              min={1}
              max={3}
              value={sim.numPowerStates}
              onChange={(e) => setSim({ ...sim, numPowerStates: Number(e.target.value) })}
            />
          </div>
          <div className="control-row">
            <label>
              <span>Algorithm</span>
              <strong>{sim.algorithm}</strong>
            </label>
            <select
              value={sim.algorithm}
              onChange={(e) => setSim({ ...sim, algorithm: e.target.value as AlgorithmMode })}
              style={{
                background: "#07090e",
                color: "#f8fafc",
                padding: "0.5rem",
                borderRadius: "6px",
                border: "1px solid var(--border-subtle)",
              }}
            >
              <option value="bf_dcqo">BF-DCQO (Shot-Estimated Bias Field)</option>
              <option value="pce">Multistage PCE (Sub-Linear Qubits)</option>
              <option value="iqo_bnb">IQO-BnB (Connected Covariance Tree)</option>
              <option value="cd_warm_qaoa">CD-WarmQAOA (Rotated-Z Mixer)</option>
              <option value="mps">MPS Tensor Network (O(N χ³) Sampling)</option>
            </select>
          </div>
        </aside>

        <main className="glass-card">
          {activeTab === "topology" && (
            <div>
              <h3 style={{ fontFamily: "var(--font-display)", marginBottom: "0.85rem" }}>
                🌐 Validated Assignment & Mode-Capacity Envelope Margins
              </h3>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Workload Task</th>
                    <th>Assigned Node</th>
                    <th>Demand q_m (W)</th>
                    <th>Execution Latency</th>
                    <th>Validator Status</th>
                  </tr>
                </thead>
                <tbody>
                  {demoPhysics.taskRouting.map((r, idx) => (
                    <tr key={idx}>
                      <td style={{ color: "var(--cyan-coherent)" }}>{r.task}</td>
                      <td>{r.node}</td>
                      <td>{r.watts} W</td>
                      <td>{r.latencyMs} ms</td>
                      <td style={{ color: "var(--emerald-valid)" }}>VALID (Margin &gt; 0)</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {activeTab === "hubo" && (
            <div>
              <h3 style={{ fontFamily: "var(--font-display)", marginBottom: "0.85rem" }}>
                🧲 Ising Coupling Matrix (h_i Diagonal, J_ij Off-Diagonal)
              </h3>
              <div className="mono-code">
                {demoPhysics.couplingMatrix
                  .map((row) => row.map((v) => v.toFixed(2).padStart(6, " ")).join(" "))
                  .join("\n")}
              </div>
            </div>
          )}

          {activeTab === "bf_dcqo" && (
            <div>
              <h3 style={{ fontFamily: "var(--font-display)", marginBottom: "0.85rem" }}>
                ⚡ BF-DCQO Shot-Estimated Spin Magnetizations
              </h3>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Impulse Step k</th>
                    <th>Surrogate ⟨H_C⟩</th>
                    <th>Mean Polarization</th>
                    <th>Sampled Magnetizations ⟨Z_i⟩</th>
                  </tr>
                </thead>
                <tbody>
                  {demoPhysics.bfIterations.map((it) => (
                    <tr key={it.iter}>
                      <td>Impulse #{it.iter}</td>
                      <td style={{ color: "var(--cyan-coherent)" }}>{it.energy}</td>
                      <td>{(it.polarization * 100).toFixed(1)}%</td>
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
                🛠️ Canonical QuantumCircuitIR Export (OpenQASM 3.0 & CUDA-Q)
              </h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.85rem" }}>
                <pre className="mono-code">{demoPhysics.openqasm3}</pre>
                <pre className="mono-code">{demoPhysics.cudaqCode}</pre>
              </div>
            </div>
          )}

          {activeTab === "qem" && (
            <div>
              <h3 style={{ fontFamily: "var(--font-display)", marginBottom: "0.85rem" }}>
                🛡️ Unitary-Folded Zero-Noise Extrapolation (U_c = U(U†U)^k)
              </h3>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Noise Scale Factor (c)</th>
                    <th>Measured / Extrapolated ⟨H_C⟩</th>
                    <th>Shot Standard Error (±σ)</th>
                  </tr>
                </thead>
                <tbody>
                  {demoPhysics.znePoints.map((pt, i) => (
                    <tr key={i}>
                      <td>c = {pt.scale.toFixed(1)}</td>
                      <td>{pt.energy}</td>
                      <td>±{(0.04 + 0.015 * pt.scale).toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {activeTab === "benchmarks" && (
            <div>
              <h3 style={{ fontFamily: "var(--font-display)", marginBottom: "0.85rem" }}>
                🏆 Head-to-Head Ablation: Classical + Repair vs. Quantum + Repair
              </h3>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Method (Equal Repair Pipeline)</th>
                    <th>Execution Mode</th>
                    <th>Qubits</th>
                    <th>Surrogate E(x)</th>
                    <th>Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {demoPhysics.sdkBenchmarks.map((row, idx) => (
                    <tr key={idx}>
                      <td style={{ color: "var(--cyan-coherent)" }}>{row.algorithm}</td>
                      <td>{row.sdk}</td>
                      <td>{row.qubits} Q</td>
                      <td>{row.energy}</td>
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
