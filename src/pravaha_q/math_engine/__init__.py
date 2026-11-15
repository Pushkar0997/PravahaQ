"""Vaisheshika 3-Body HUBO & Ising Spin-Glass Mathematical Compiler."""

from pravaha_q.math_engine.feasibility_repair import (
    DecodedInfrastructureSchedule,
    FeasibilityRepairEngine,
)
from pravaha_q.math_engine.hubo_builder import HUBOPolynomial, VaisheshikaHUBOCompiler
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass, hubo_to_ising
from pravaha_q.math_engine.spectral_analyzer import SpectralReport, analyze_ising_spectrum
from pravaha_q.math_engine.unbalanced_penalties import (
    InequalityConstraintSpec,
    compile_unbalanced_inequality_penalty,
    evaluate_unbalanced_penalty_curve,
)

__all__ = [
    "DecodedInfrastructureSchedule",
    "FeasibilityRepairEngine",
    "HUBOPolynomial",
    "InequalityConstraintSpec",
    "IsingSpinGlass",
    "SpectralReport",
    "VaisheshikaHUBOCompiler",
    "analyze_ising_spectrum",
    "compile_unbalanced_inequality_penalty",
    "evaluate_unbalanced_penalty_curve",
    "hubo_to_ising",
]
