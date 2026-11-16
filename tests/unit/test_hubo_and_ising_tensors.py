"""Unit tests verifying exact algebraic identity E_HUBO(x) == <x| H_C |x> across all 2^N states."""

from __future__ import annotations

import numpy as np

from pravaha_q.math_engine.hubo_builder import HUBOPolynomial
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass, hubo_to_ising
from pravaha_q.math_engine.spectral_analyzer import analyze_ising_spectrum


def test_hand_worked_3qubit_walkthrough_exact_truth_table() -> None:
    """
    Verify the exact 3-qubit worked example from docs/01_VISUAL_MATH_WALKTHROUGH.md:
      E(x_1, x_2, x_3) = -4x_1 + 2x_2 - 3x_3 - 2x_1 x_2 + 2x_1 x_3 + 8x_1 x_2 x_3
    """
    hubo = HUBOPolynomial(num_variables=3)
    hubo.add_linear(0, -4.0)
    hubo.add_linear(1, 2.0)
    hubo.add_linear(2, -3.0)
    hubo.add_quadratic(0, 1, -2.0)
    hubo.add_quadratic(0, 2, 2.0)
    hubo.add_cubic(0, 1, 2, 8.0)

    ising = hubo_to_ising(hubo)

    assert np.isclose(ising.constant_offset, -1.5)
    assert np.isclose(ising.h_linear[0], 1.0)
    assert np.isclose(ising.h_linear[1], -1.5)
    assert 2 not in ising.h_linear
    assert np.isclose(ising.j_quadratic[(0, 1)], 0.5)
    assert np.isclose(ising.j_quadratic[(0, 2)], 1.5)
    assert np.isclose(ising.j_quadratic[(1, 2)], 1.0)
    assert np.isclose(ising.k_cubic[(0, 1, 2)], -1.0)

    expected_energies = {
        "000": 0.0,
        "001": -3.0,
        "010": 2.0,
        "011": -1.0,
        "100": -4.0,
        "101": -5.0,
        "110": -4.0,
        "111": 3.0,
    }
    for bstr, exp_e in expected_energies.items():
        assert np.isclose(hubo.evaluate(bstr), exp_e, atol=1e-12)
        assert np.isclose(ising.evaluate_bitstring(bstr), exp_e, atol=1e-12)

    spec = analyze_ising_spectrum(ising)
    assert spec.ground_state_bitstring == "101"
    assert np.isclose(spec.ground_energy, -5.0)


def test_full_2_to_n_hubo_ising_isomorphism(
    small_hubo: HUBOPolynomial, small_ising: IsingSpinGlass
) -> None:
    """Verify Theorem 1 across all 2^8 = 256 states of the 8-qubit coupled problem."""
    n = small_ising.num_qubits
    diag = small_ising.diagonal_vector()
    for idx in range(1 << n):
        bstr = format(idx, f"0{n}b")
        e_hubo = small_hubo.evaluate(bstr)
        e_ising = small_ising.evaluate_bitstring(bstr)
        assert np.isclose(e_hubo, e_ising, atol=1e-10)
        assert np.isclose(e_hubo, diag[idx], atol=1e-10)
