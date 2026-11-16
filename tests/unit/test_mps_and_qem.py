"""Unit tests for MPS Tensor Network surrogate and ZNE/TREX/CVaR error mitigation."""

from __future__ import annotations

from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass
from pravaha_q.mitigation.cvar_tail_estimator import compute_cvar_expectation
from pravaha_q.mitigation.readout_mitigation import TREXReadoutMitigator
from pravaha_q.mitigation.zne_folding import richardson_zero_noise_extrapolation
from pravaha_q.tensor_network.mps_simulator import MPSSimulator


def test_mps_tensor_network_surrogate(small_ising: IsingSpinGlass) -> None:
    mps = MPSSimulator(max_bond_dim=16, seed=42)
    res = mps.solve(small_ising)
    assert res.num_qubits == small_ising.num_qubits
    assert len(res.best_bitstring) == small_ising.num_qubits


def test_zne_trex_and_cvar_mitigation(small_ising: IsingSpinGlass) -> None:
    counts = {
        "0" * small_ising.num_qubits: 400,
        "1" * small_ising.num_qubits: 400,
    }
    mitigated = TREXReadoutMitigator().mitigate_counts(
        counts, small_ising.num_qubits
    )
    assert len(mitigated) > 0

    cvar_full = compute_cvar_expectation(
        mitigated, small_ising.evaluate_bitstring, alpha=1.0
    )
    cvar_tail = compute_cvar_expectation(
        mitigated, small_ising.evaluate_bitstring, alpha=0.25
    )
    assert cvar_tail <= cvar_full + 1e-9

    # Linear noise model y(c) = -10.0 + 2.0 * c should extrapolate to exactly -10.0 at c=0
    zne_val = richardson_zero_noise_extrapolation(
        scale_factors=(1.0, 3.0, 5.0),
        measured_expectations=(-8.0, -4.0, 0.0),
    )
    assert abs(zne_val - (-10.0)) < 1e-6
