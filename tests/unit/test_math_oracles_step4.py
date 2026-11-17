"""
Independent Reference & Matrix Oracle Verification Suite (Step 4)
=================================================================
Tests each of the 7 mathematical components against independent matrix oracles
that would have caught the original bugs:
  1. Pauli expectations vs explicit 2^n x 2^n Kronecker matrix multiplication
     on random complex statevectors (including odd-Y and mixed Pauli strings).
  2. AGP coefficient alpha_1*(lambda) via sparse Pauli commutator algebra vs
     explicit 2^N x 2^N matrix trace oracle across random cubic Hamiltonians.
  3. 3-body counterdiabatic gate apply_three_qubit_yzz vs `scipy.linalg.expm`
     of -i * theta * (YZZ + ZYZ + ZZY).
  4. Warm-start rotated-Z mixer preserving |psi_0> = bigotimes_i Ry(theta_i)|0>
     up to pure global phase.
  5. Spectral gap & degeneracy invariance under large identity shifts H -> H + c*I.
  6. Rigorous SDP lower bound L <= E* across random 2-body and 3-body Hamiltonians.
  7. Connected covariance Cov(Z_i, Z_j) == 0 on arbitrary unentangled product states.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from pravaha_q.algorithms.agp_commutator import (
    compute_agp_alpha1_matrix_oracle,
    compute_agp_alpha1_pauli_algebra,
)
from pravaha_q.algorithms.bf_dcqo import (
    apply_single_qubit_ry,
    apply_three_qubit_yzz,
)
from pravaha_q.algorithms.cd_warm_qaoa import (
    apply_warmstart_mixer_layer,
    prepare_warmstart_state,
)
from pravaha_q.algorithms.iqo_branch_and_bound import (
    compute_connected_covariance_matrix,
)
from pravaha_q.encodings.pauli_correlation_encoding import (
    MultistagePCEEncoder,
    evaluate_pauli_string_matrix_oracle,
    generate_pauli_correlator_basis,
)
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass
from pravaha_q.math_engine.spectral_analyzer import analyze_ising_spectrum
from pravaha_q.warmstart.sdp_goemans_williamson import solve_gw_sdp_relaxation


def test_1_pauli_expectations_match_explicit_kronecker_matrix_oracle() -> None:
    """Compare fast bitmask Pauli expectations against explicit matrix multiplication."""
    n = 4
    rng = np.random.default_rng(101)
    raw = rng.normal(size=1 << n) + 1.0j * rng.normal(size=1 << n)
    state = raw / np.linalg.norm(raw)

    correlators = generate_pauli_correlator_basis(
        num_qubits=n, num_variables=42, max_body_k=3
    )
    masks = [MultistagePCEEncoder._compile_pauli_mask(c, n) for c in correlators]
    fast_exps = MultistagePCEEncoder._evaluate_all_correlators(state, masks)

    for corr, fast_val in zip(correlators, fast_exps):
        oracle_val = evaluate_pauli_string_matrix_oracle(state, corr, n)
        assert np.isclose(fast_val, oracle_val, rtol=0.0, atol=1e-13), (
            f"Mismatch on Pauli correlator {corr}: fast={fast_val}, oracle={oracle_val}"
        )


def test_2_agp_coefficient_matches_explicit_matrix_trace_oracle() -> None:
    """Compare sparse Pauli commutator AGP alpha_1* against 2^N matrix trace oracle."""
    rng = np.random.default_rng(202)
    for _ in range(5):
        n = 5
        h_lin = {i: float(rng.uniform(-2.0, 2.0)) for i in range(n)}
        j_quad = {
            (i, j): float(rng.uniform(-1.5, 1.5))
            for i in range(n)
            for j in range(i + 1, n)
        }
        k_cub = {
            (0, 1, 2): float(rng.uniform(-1.2, 1.2)),
            (1, 2, 4): float(rng.uniform(-1.2, 1.2)),
        }
        ising = IsingSpinGlass(
            num_qubits=n,
            constant_offset=3.5,
            h_linear=h_lin,
            j_quadratic=j_quad,
            k_cubic=k_cub,
        )
        bias = {i: float(rng.uniform(-0.5, 0.5)) for i in range(n)}
        lam = float(rng.uniform(0.15, 0.85))

        a_pauli, g1_p, g2_p = compute_agp_alpha1_pauli_algebra(ising, bias, lam)
        a_mat, g1_m, g2_m = compute_agp_alpha1_matrix_oracle(ising, bias, lam)

        assert np.isclose(g1_p, g1_m, rtol=1e-12, atol=1e-12)
        assert np.isclose(g2_p, g2_m, rtol=1e-12, atol=1e-12)
        assert np.isclose(a_pauli, a_mat, rtol=1e-12, atol=1e-12)


def test_3_three_body_yzz_gate_matches_matrix_exponential() -> None:
    """Verify apply_three_qubit_yzz against scipy.linalg.expm(-i theta (YZZ + ZYZ + ZZY))."""
    Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    op_3body = (
        np.kron(np.kron(Y, Z), Z)
        + np.kron(np.kron(Z, Y), Z)
        + np.kron(np.kron(Z, Z), Y)
    )
    theta = 0.37
    u_exact = expm(-1.0j * theta * op_3body)

    rng = np.random.default_rng(303)
    raw = rng.normal(size=8) + 1.0j * rng.normal(size=8)
    psi_0 = raw / np.linalg.norm(raw)

    psi_tensor = apply_three_qubit_yzz(
        psi_0.reshape(2, 2, 2), 0, 1, 2, theta, num_qubits=3
    ).reshape(-1)
    psi_exact = u_exact @ psi_0

    assert np.allclose(psi_tensor, psi_exact, rtol=0.0, atol=1e-14)


def test_4_warmstart_rotated_z_mixer_preserves_initial_state_up_to_phase() -> None:
    """Verify U_M(beta) |psi_0> = e^{i N beta} |psi_0> for any warm-start angles."""
    angles = [0.32, 1.15, 2.41, 0.78]
    beta = 0.63
    psi_0 = prepare_warmstart_state(angles)
    psi_after = apply_warmstart_mixer_layer(psi_0.copy(), angles, beta)

    vec_0 = psi_0.reshape(-1)
    vec_after = psi_after.reshape(-1)

    overlap = np.vdot(vec_0, vec_after)
    assert np.isclose(abs(overlap), 1.0, rtol=0.0, atol=1e-14)
    expected_phase = np.exp(1.0j * len(angles) * beta)
    assert np.allclose(vec_after, expected_phase * vec_0, rtol=0.0, atol=1e-14)


def test_5_spectral_analysis_shift_invariance_under_large_identity_offset() -> None:
    """Adding c*I (even c = 10^7) must not change spectral gap or ground degeneracy."""
    ising_base = IsingSpinGlass(
        num_qubits=4,
        constant_offset=0.0,
        h_linear={0: 0.5, 1: -0.5},
        j_quadratic={(0, 1): 1.0, (2, 3): -1.0},
    )
    ising_shifted = IsingSpinGlass(
        num_qubits=4,
        constant_offset=1.0e7,
        h_linear=dict(ising_base.h_linear),
        j_quadratic=dict(ising_base.j_quadratic),
    )

    rep_base = analyze_ising_spectrum(ising_base)
    rep_shift = analyze_ising_spectrum(ising_shifted)

    assert rep_base.ground_state_degeneracy == rep_shift.ground_state_degeneracy == 2
    assert np.isclose(rep_base.spectral_gap, rep_shift.spectral_gap, rtol=0.0, atol=1e-8)
    assert rep_base.ground_state_bitstring == rep_shift.ground_state_bitstring


def test_6_rigorous_lower_bound_holds_with_cubic_terms() -> None:
    """Exhaustively verify sdp_lower_bound <= E* across random cubic Hamiltonians."""
    rng = np.random.default_rng(606)
    for _ in range(15):
        n = 5
        ising = IsingSpinGlass(
            num_qubits=n,
            constant_offset=float(rng.uniform(-5.0, 5.0)),
            h_linear={i: float(rng.uniform(-2.0, 2.0)) for i in range(n)},
            j_quadratic={
                (i, j): float(rng.uniform(-1.5, 1.5))
                for i in range(n)
                for j in range(i + 1, n)
            },
            k_cubic={
                (0, 1, 2): float(rng.uniform(-3.0, 3.0)),
                (1, 3, 4): float(rng.uniform(-3.0, 3.0)),
                (0, 2, 4): float(rng.uniform(-3.0, 3.0)),
            },
        )
        spec = analyze_ising_spectrum(ising)
        sdp_res = solve_gw_sdp_relaxation(ising, seed=42)
        assert sdp_res.sdp_lower_bound <= spec.ground_energy + 1e-7


def test_7_product_states_have_strictly_zero_connected_covariance() -> None:
    """Any unentangled product state must have zero off-diagonal connected covariance."""
    n = 5
    state = np.zeros(1 << n, dtype=np.complex128)
    state[0] = 1.0 + 0.0j
    psi = state.reshape((2,) * n)

    for q, theta in enumerate([0.25, -0.91, 1.42, 2.15, -0.44]):
        psi = apply_single_qubit_ry(psi, q, theta)

    probs = np.abs(psi.reshape(-1)) ** 2
    cov = compute_connected_covariance_matrix(probs, n)
    off_diag = cov - np.diag(np.diag(cov))
    assert np.max(np.abs(off_diag)) < 1e-14
