"""Full multi-algorithm ablation and benchmarking suite runner."""

from __future__ import annotations

import json
from pathlib import Path

from pravaha_q.domain.problem_generator import generate_coupled_problem
from pravaha_q.math_engine.hubo_builder import VaisheshikaHUBOCompiler
from pravaha_q.math_engine.ising_spin_glass import hubo_to_ising
from pravaha_q.solvers.classical_baselines import ExactSpectralSolver
from pravaha_q.solvers.qiskit_adapter import QiskitBFDCQOSolver


def run_full_ablation_suite(
    output_json: str | Path | None = None,
) -> dict[str, float]:
    """Execute ablation benchmarks across problem instances and export telemetry."""
    prob = generate_coupled_problem(num_nodes=2, num_tasks=3, seed=42)
    hubo = VaisheshikaHUBOCompiler().compile(prob)
    ising = hubo_to_ising(hubo)

    exact = ExactSpectralSolver().solve(ising)
    bf = QiskitBFDCQOSolver().solve(ising)

    summary = {
        "exact_ground_energy": exact.best_energy,
        "bf_dcqo_energy": bf.best_energy,
        "approximation_ratio": bf.approximation_ratio,
        "execution_time_ms": bf.execution_time_ms,
    }

    if output_json is not None:
        p = Path(output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary
