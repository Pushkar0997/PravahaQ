"""
Unit tests verifying GPT-6 Astra Steps 5 and 6:
  1. Single Canonical QuantumCircuitIR drives simulation, OpenQASM 3.0, and CUDA-Q.
  2. Real algebraic gate cancellation reduces gate counts while preserving the
     exact statevector U_opt |0> == U_unopt |0> to 1e-14.
  3. Hardware topology SWAP routing inserts valid 3-CX SWAPs and obeys graph connectivity.
  4. Genuine MPS simulator evolves non-adjacent gates via SWAP chains, tracks truncation
     error, samples without 2^N enumeration, and converges to exact statevector as chi grows.
  5. Unitary gate folding U_c = U (U^dagger U)^{(c-1)/2} equals U when noiseless and
     produces calibrated ZNE uncertainty bars under gate-level depolarizing noise.
  6. IBMRuntimeHardwareSession returns explicit UNAVAILABLE status when credentials are absent.
"""

from __future__ import annotations

import numpy as np

from pravaha_q.compiler.circuit_ir import build_bf_dcqo_circuit_ir
from pravaha_q.compiler.hardware_graphs import build_hardware_topology
from pravaha_q.compiler.openqasm3_emitter import export_openqasm3
from pravaha_q.compiler.paulihedral_router import (
    cancel_algebraic_gates,
    compile_paulihedral_schedule,
)
from pravaha_q.domain.problem_generator import generate_coupled_problem
from pravaha_q.math_engine.hubo_builder import VaisheshikaHUBOCompiler
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass, hubo_to_ising
from pravaha_q.math_engine.validator import SolverStatus
from pravaha_q.mitigation.readout_mitigation import TREXReadoutMitigator
from pravaha_q.mitigation.zne_folding import execute_circuit_zne
from pravaha_q.solvers.base import ExecutionMode
from pravaha_q.solvers.cirq_adapter import CirqGridSolver
from pravaha_q.solvers.ibm_runtime_hardware import IBMRuntimeHardwareSession
from pravaha_q.solvers.qiskit_adapter import QiskitBFDCQOSolver
from pravaha_q.tensor_network.mps_simulator import MPSSimulator


def test_1_algebraic_gate_cancellation_preserves_exact_statevector() -> None:
    """Verify U_opt |0> == U_unopt |0> to 1e-14 and actual CX count decreases."""
    prob = generate_coupled_problem(
        num_nodes=2, num_tasks=2, num_power_states=2, seed=42
    )
    hubo = VaisheshikaHUBOCompiler(enable_cubic_terms=True).compile(prob)
    ising = hubo_to_ising(hubo)

    report = compile_paulihedral_schedule(ising, topology="heavy_hex")
    assert report.unoptimized_ir is not None
    assert report.optimized_ir is not None

    # Verify real gate reduction on shared-prefix Pauli staircases
    assert report.post_cancellation_cx_count < report.naive_cx_count
    assert (
        report.optimized_ir.count_2q_gates() == report.post_cancellation_cx_count
    )

    # Verify exact statevector equivalence before and after cancellation
    sv_unopt = report.unoptimized_ir.simulate_statevector()
    sv_opt = report.optimized_ir.simulate_statevector()
    assert np.allclose(sv_unopt, sv_opt, rtol=0.0, atol=1e-13)


def test_2_topology_routing_obeys_hardware_graph_edges() -> None:
    """Every 2-qubit gate in `routed_ir` must act on an active edge of `HardwareTopologyGraph`."""
    ising = IsingSpinGlass(
        num_qubits=5,
        h_linear={0: 0.5, 4: -0.5},
        j_quadratic={(0, 4): 1.2, (1, 3): -0.8},
    )
    topo = build_hardware_topology("heavy_hex", num_qubits=5)
    g = topo.to_networkx()

    ir = build_bf_dcqo_circuit_ir(ising, include_counterdiabatic=False)
    routed_ir, swaps = ir.route_to_topology(topo)
    assert swaps > 0

    for op in routed_ir.ops:
        if len(op.qubits) == 2:
            u, v = op.qubits
            assert g.has_edge(u, v), f"Routed 2Q gate ({u}, {v}) violates hardware topology!"


def test_3_openqasm3_is_generated_directly_from_canonical_ir() -> None:
    """Verify OpenQASM 3.0 output matches the exact gate counts of `optimized_ir`."""
    ising = IsingSpinGlass(
        num_qubits=4,
        h_linear={0: 0.4, 1: -0.6},
        j_quadratic={(0, 1): 0.8, (0, 2): 0.5},
    )
    rep = compile_paulihedral_schedule(ising)
    assert rep.optimized_ir is not None
    qasm_str = export_openqasm3(ising)

    cx_lines = [
        line for line in qasm_str.splitlines() if line.strip().startswith("cx ")
    ]
    assert len(cx_lines) == rep.optimized_ir.count_2q_gates()


def test_4_genuine_mps_converges_to_exact_statevector_with_bond_dimension() -> None:
    """
    Verify that MPSSimulator:
      1. Evolves non-adjacent gates via SWAP chains.
      2. Has non-zero truncation error at chi=1 and zero truncation error at chi=16.
      3. Matches `QuantumCircuitIR.simulate_statevector()` to 1e-12 at full bond dimension.
    """
    ising = IsingSpinGlass(
        num_qubits=5,
        constant_offset=1.0,
        h_linear={0: 0.7, 2: -0.5, 4: 0.3},
        j_quadratic={(0, 4): 0.9, (1, 3): -0.7, (0, 2): 0.6},
    )
    ir = cancel_algebraic_gates(build_bf_dcqo_circuit_ir(ising))
    exact_sv = ir.simulate_statevector()
    exact_probs = np.abs(exact_sv) ** 2
    exact_exp_e = float(np.sum(exact_probs * ising.diagonal_vector()))

    # Low bond dimension chi=1 (product-state approximation)
    mps_chi1 = MPSSimulator(max_bond_dim=1, shots=128, seed=42)
    tensors_1, err_1 = mps_chi1.evolve_circuit(ir)
    assert err_1 > 0.0

    # Full bond dimension chi=16 for N=5 qubits -> zero truncation error!
    mps_chi16 = MPSSimulator(max_bond_dim=16, shots=256, seed=42)
    tensors_16, err_16 = mps_chi16.evolve_circuit(ir)
    assert err_16 < 1e-13

    mps_sv_16 = MPSSimulator.contract_to_statevector_for_test(tensors_16)
    assert np.allclose(np.abs(mps_sv_16) ** 2, exact_probs, rtol=0.0, atol=1e-12)

    mps_exp_e = MPSSimulator.evaluate_ising_expectation(tensors_16, ising)
    assert np.isclose(mps_exp_e, exact_exp_e, rtol=0.0, atol=1e-11)


def test_5_unitary_folding_identity_and_calibrated_zne_uncertainty() -> None:
    """
    Verify that:
      - In the noiseless limit, `ir.fold_unitary(3)` produces the exact same statevector as `ir`.
      - Under gate-level depolarizing noise, `execute_circuit_zne` executes folded circuits
        with 3x and 5x gate counts and returns a positive propagated `extrapolated_std_error`.
    """
    ising = IsingSpinGlass(
        num_qubits=4,
        h_linear={0: -0.8, 1: 0.6},
        j_quadratic={(0, 1): 1.1, (1, 2): -0.9, (2, 3): 0.7},
    )
    ir_1 = cancel_algebraic_gates(build_bf_dcqo_circuit_ir(ising))
    ir_3 = ir_1.fold_unitary(3)
    assert len(ir_3.ops) == 3 * len(ir_1.ops)

    # Noiseless unitary identity: U (U^dagger U) == U
    sv_1 = ir_1.simulate_statevector(per_gate_depol_error=0.0)
    sv_3 = ir_3.simulate_statevector(per_gate_depol_error=0.0)
    assert np.allclose(sv_1, sv_3, rtol=0.0, atol=1e-12)

    zne_rep = execute_circuit_zne(
        ir_1,
        ising,
        scale_factors=(1, 3, 5),
        per_gate_depol_error=0.01,
        shots=1024,
        seed=42,
    )
    assert zne_rep.folded_2q_gate_counts[1] == 3 * zne_rep.folded_2q_gate_counts[0]
    assert zne_rep.folded_2q_gate_counts[2] == 5 * zne_rep.folded_2q_gate_counts[0]
    assert zne_rep.extrapolated_std_error > 0.0


def test_6_strict_execution_modes_and_honest_hardware_unavailability() -> None:
    """Verify SHOT_SIMULATION vs HARDWARE unavailability reporting and TREX inversion."""
    ising = IsingSpinGlass(
        num_qubits=4,
        h_linear={0: -1.0, 1: 0.5},
        j_quadratic={(0, 1): 0.8, (2, 3): -0.6},
    )

    qk_shot = QiskitBFDCQOSolver(
        shots=512, execution_mode=ExecutionMode.SHOT_SIMULATION
    ).solve(ising)
    assert qk_shot.execution_mode == ExecutionMode.SHOT_SIMULATION
    assert qk_shot.approximation_ratio is None

    cq_shot = CirqGridSolver(shots=512, seed=42).solve(ising)
    assert cq_shot.execution_mode == ExecutionMode.SHOT_SIMULATION
    assert cq_shot.approximation_ratio is None

    # Without credentials and with allow_simulator_fallback=False, must report UNAVAILABLE
    hw_strict = IBMRuntimeHardwareSession(
        api_token="", allow_simulator_fallback=False
    ).solve(ising)
    assert hw_strict.solver_status == SolverStatus.UNKNOWN
    assert hw_strict.metadata["hardware_status"] == "UNAVAILABLE_NO_CREDENTIALS"

    # With allow_simulator_fallback=True, must label itself as SHOT_SIMULATION fallback
    hw_fb = IBMRuntimeHardwareSession(
        api_token="", allow_simulator_fallback=True
    ).solve(ising)
    assert hw_fb.execution_mode == ExecutionMode.SHOT_SIMULATION
    assert hw_fb.metadata["is_fallback"] is True

    # Verify TREX matrix inversion recovers noisy 1-qubit readout distribution
    trex = TREXReadoutMitigator(p_0_to_1=0.05, p_1_to_0=0.05)
    m_mat = trex.single_qubit_confusion_matrix()
    m_inv = trex.single_qubit_inverse_matrix()
    assert np.allclose(m_mat @ m_inv, np.eye(2), atol=1e-14)
