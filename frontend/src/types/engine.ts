export type AlgorithmMode = "bf_dcqo" | "pce" | "iqo_bnb" | "cd_warm_qaoa" | "mps";

export type SDKTarget = "qiskit" | "pennylane" | "cirq" | "mps" | "ibm_runtime";

export type HardwareTopology = "heavy_hex" | "planar_grid" | "all_to_all";

export interface SimulationState {
  numNodes: number;
  numTasks: number;
  numPowerStates: number;
  algorithm: AlgorithmMode;
  sdk: SDKTarget;
  topology: HardwareTopology;
  pceQubits: number;
  unbalancedLambda1: number;
  unbalancedLambda2: number;
  cvarAlpha: number;
  enableCubic3Body: boolean;
}

export interface ComputedPhysicsPayload {
  totalBinaryVars: number;
  qubitsExecuted: number;
  compressionRatio: number;
  groundEnergy: number;
  foundEnergy: number;
  approximationRatio: number;
  spectralGap: number;
  frustrationIndex: number;
  paulihedralCxSavedPct: number;
  bestBitstring: string;
  couplingMatrix: number[][];
  cubicTriads: { triad: string; weight: number }[];
  bfIterations: { iter: number; energy: number; polarization: number; mags: number[] }[];
  znePoints: { scale: number; energy: number }[];
  openqasm3: string;
  cudaqCode: string;
  taskRouting: { task: string; node: string; watts: number; latencyMs: number }[];
  sdkBenchmarks: {
    algorithm: string;
    sdk: string;
    qubits: number;
    energy: number;
    ar: number;
    latencyMs: number;
  }[];
}
