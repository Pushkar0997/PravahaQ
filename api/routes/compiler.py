"""Endpoints for Vaisheshika 3-Body HUBO, Ising Spin Glass, OpenQASM 3.0 & CUDA-Q."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter

from api.schemas import CompileRequest
from pravaha_q.compiler.cudaq_qir_exporter import export_cudaq_kernel_source
from pravaha_q.compiler.openqasm3_emitter import export_openqasm3
from pravaha_q.compiler.paulihedral_router import compile_paulihedral_schedule
from pravaha_q.config import UnbalancedPenaltyConfig
from pravaha_q.domain.problem_generator import generate_coupled_problem
from pravaha_q.infrastructure.content_addressable_cache import (
    compute_hamiltonian_sha256,
)
from pravaha_q.math_engine.hubo_builder import VaisheshikaHUBOCompiler
from pravaha_q.math_engine.ising_spin_glass import hubo_to_ising
from pravaha_q.math_engine.spectral_analyzer import analyze_ising_spectrum
from pravaha_q.math_engine.unbalanced_penalties import (
    evaluate_unbalanced_penalty_curve,
)

router = APIRouter(prefix="/compiler", tags=["2. Vaisheshika HUBO & Quantum Compiler"])


@router.post("/inspect")
def compile_and_inspect_hamiltonian(req: CompileRequest) -> dict[str, Any]:
    """
    Compile problem into a 3-Local HUBO and Ising Spin Glass, run Paulihedral
    gate cancellation, and emit OpenQASM 3.0 + NVIDIA CUDA-Q kernels.
    """
    prob = generate_coupled_problem(
        num_nodes=req.num_nodes,
        num_tasks=req.num_tasks,
        num_power_states=req.num_power_states,
        seed=req.seed,
    )
    pen_cfg = UnbalancedPenaltyConfig(
        lambda1_linear=req.unbalanced_lambda1,
        lambda2_quadratic=req.unbalanced_lambda2,
        equality_weight=req.equality_penalty,
    )
    hubo = VaisheshikaHUBOCompiler(
        penalty_config=pen_cfg, enable_cubic_terms=req.enable_cubic_3body
    ).compile(prob)
    ising = hubo_to_ising(hubo)

    pauli_rep = compile_paulihedral_schedule(
        ising,
        topology=req.topology,
        include_counterdiabatic=req.include_counterdiabatic,
    )
    qasm3_code = export_openqasm3(
        ising,
        gamma=req.gamma,
        beta=req.beta,
        include_counterdiabatic=req.include_counterdiabatic,
    )
    cudaq_code, qir_profile = export_cudaq_kernel_source(
        ising, gamma=req.gamma, beta=req.beta
    )

    spec_data: dict[str, Any] | None = None
    if ising.num_qubits <= 16:
        spec_data = asdict(analyze_ising_spectrum(ising))

    slack_grid = [round(-0.6 + 0.1 * i, 2) for i in range(17)]
    penalty_curve = evaluate_unbalanced_penalty_curve(
        slack_grid, lambda1=req.unbalanced_lambda1, lambda2=req.unbalanced_lambda2
    )

    return {
        "hamiltonian_sha256": compute_hamiltonian_sha256(ising),
        "num_qubits": ising.num_qubits,
        "constant_offset": round(ising.constant_offset, 5),
        "spectral_scale": round(ising.spectral_scale, 5),
        "h_linear": {str(k): round(v, 5) for k, v in ising.h_linear.items()},
        "j_quadratic": {
            f"{u},{v}": round(val, 5) for (u, v), val in ising.j_quadratic.items()
        },
        "k_cubic": {
            f"{u},{v},{w}": round(val, 5) for (u, v, w), val in ising.k_cubic.items()
        },
        "coupling_matrix": ising.to_coupling_matrix().tolist(),
        "pauli_strings": ising.to_pauli_list()[:32],
        "paulihedral_report": asdict(pauli_rep),
        "qir_resource_profile": asdict(qir_profile),
        "openqasm3_source": qasm3_code,
        "cudaq_kernel_source": cudaq_code,
        "spectral_report": spec_data,
        "unbalanced_penalty_curve": {
            "slack_grid": slack_grid,
            "penalty_values": [round(v, 4) for v in penalty_curve],
        },
    }
