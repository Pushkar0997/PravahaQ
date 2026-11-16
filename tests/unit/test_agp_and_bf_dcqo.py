"""Unit tests for analytical Adiabatic Gauge Potentials (AGP) and BF-DCQO."""

from __future__ import annotations

from pravaha_q.algorithms.agp_commutator import compute_analytical_agp_schedule
from pravaha_q.algorithms.bf_dcqo import BFDCQOEngine
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass
from pravaha_q.math_engine.spectral_analyzer import analyze_ising_spectrum


def test_analytical_agp_schedule_properties(small_ising: IsingSpinGlass) -> None:
    sched = compute_analytical_agp_schedule(small_ising, lambda_point=0.5)
    assert sched.alpha1_coeff < 0.0
    assert sched.gamma1_norm > 0.0
    assert sched.gamma2_norm > 0.0
    assert len(sched.y_1body_angles) > 0


def test_bf_dcqo_converges_near_ground_state(small_ising: IsingSpinGlass) -> None:
    spec = analyze_ising_spectrum(small_ising)
    engine = BFDCQOEngine(shots=1024, seed=42)
    res = engine.run(small_ising)

    assert res.num_qubits == small_ising.num_qubits
    assert res.iterations_executed >= 1
    assert res.best_energy <= spec.max_energy
    # Approximation ratio should be high (> 0.85)
    ar = (spec.max_energy - res.best_energy) / (spec.max_energy - spec.ground_energy)
    assert ar > 0.85
