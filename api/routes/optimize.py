"""Hardened optimization endpoint with Resource Admission Guard, Real ZNE & Independent Validation."""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from api.schemas import (
    OptimizeRequest,
    OptimizeResponse,
    OptimizeSchedulePayload,
)
from pravaha_q.algorithms.bf_dcqo import BFDCQOEngine
from pravaha_q.algorithms.cd_warm_qaoa import CDWarmQAOASolver
from pravaha_q.algorithms.iqo_branch_and_bound import IQOBranchAndBoundSolver
from pravaha_q.benchmarks.metrics import compute_approximation_ratio
from pravaha_q.compiler.paulihedral_router import compile_paulihedral_schedule
from pravaha_q.config import BFDCQOConfig, UnbalancedPenaltyConfig
from pravaha_q.domain.problem_generator import generate_coupled_problem
from pravaha_q.encodings.pauli_correlation_encoding import MultistagePCEEncoder
from pravaha_q.infrastructure.admission_guard import (
    AlgorithmType,
    CompatibilityError,
    ResourceAdmissionError,
    check_execution_admission,
)
from pravaha_q.infrastructure.async_job_fsm import JobState, QuantumJobRecord
from pravaha_q.infrastructure.content_addressable_cache import (
    compute_hamiltonian_sha256,
)
from pravaha_q.infrastructure.observability import GLOBAL_TELEMETRY
from pravaha_q.math_engine.feasibility_repair import FeasibilityRepairEngine
from pravaha_q.math_engine.hubo_builder import VaisheshikaHUBOCompiler
from pravaha_q.math_engine.ising_spin_glass import hubo_to_ising
from pravaha_q.math_engine.spectral_analyzer import analyze_ising_spectrum
from pravaha_q.mitigation.cvar_tail_estimator import compute_cvar_expectation
from pravaha_q.mitigation.readout_mitigation import TREXReadoutMitigator
from pravaha_q.mitigation.zne_folding import execute_circuit_zne
from pravaha_q.solvers.base import ExecutionMode
from pravaha_q.solvers.classical_baselines import (
    ExactSpectralSolver,
    GreedyOperationalSolver,
    SimulatedAnnealingSolver,
    StructuredEnumerationSolver,
)
from pravaha_q.tensor_network.mps_simulator import MPSSimulator

router = APIRouter(prefix="/optimize", tags=["3. Hybrid Quantum Optimization Engine"])


@router.post("/run", response_model=OptimizeResponse)
def execute_optimization(req: OptimizeRequest) -> OptimizeResponse:
    """
    Execute the PravahaQ pipeline with pre-execution admission checks, honest
    ExecutionMode labeling, real unitary-folded ZNE, and independent MILP validation.
    """
    start_t = time.perf_counter()
    prob = generate_coupled_problem(
        num_nodes=req.num_nodes,
        num_tasks=req.num_tasks,
        num_power_states=req.num_power_states,
        seed=req.seed,
        capacity_scale=req.capacity_scale,
    )

    # Step 8: Pre-execution resource & compatibility admission guard
    try:
        admission = check_execution_admission(
            problem=prob,
            algorithm=req.algorithm,
            backend=req.sdk,
            pce_qubits=req.pce_qubits,
            mps_bond_dim=req.mps_bond_dim,
        )
    except (CompatibilityError, ResourceAdmissionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    job = QuantumJobRecord(algorithm=req.algorithm.value, sdk=req.sdk.value)
    job.transition_to(JobState.COMPILING_HUBO)

    pen_cfg = UnbalancedPenaltyConfig(
        lambda1_linear=req.unbalanced_lambda1,
        lambda2_quadratic=req.unbalanced_lambda2,
        equality_weight=req.equality_penalty,
    )
    hubo = VaisheshikaHUBOCompiler(
        penalty_config=pen_cfg, enable_cubic_terms=req.enable_cubic_3body
    ).compile(prob)
    ising = hubo_to_ising(hubo)
    h_sha = compute_hamiltonian_sha256(ising)

    job.transition_to(JobState.EXECUTING_QUANTUM)
    algo = admission.algorithm
    num_qubits_used = admission.num_qubits_allocated
    iter_telemetry: list[dict[str, Any]] = []
    exec_mode = ExecutionMode.SHOT_SIMULATION

    if algo == AlgorithmType.PCE:
        exec_mode = ExecutionMode.IDEAL_STATEVECTOR
        pce_res = MultistagePCEEncoder(
            num_qubits=num_qubits_used, seed=req.seed
        ).solve(ising)
        raw_best_bits = pce_res.best_bitstring
        counts = {raw_best_bits: req.shots}
        for idx, l_val in enumerate(pce_res.loss_history[::5]):
            iter_telemetry.append(
                {"iteration": idx + 1, "expectation_energy": round(l_val, 5)}
            )
    elif algo == AlgorithmType.IQO_BNB:
        exec_mode = ExecutionMode.IDEAL_STATEVECTOR
        iqo_res = IQOBranchAndBoundSolver(seed=req.seed).solve(ising)
        raw_best_bits = iqo_res.best_bitstring
        counts = {raw_best_bits: req.shots}
        for idx, step in enumerate(iqo_res.reduction_trace):
            iter_telemetry.append(
                {
                    "iteration": idx + 1,
                    "remaining_qubits": step.remaining_qubits,
                    "correlation_magnitude": step.correlation_magnitude,
                }
            )
    elif algo == AlgorithmType.CD_WARM_QAOA:
        exec_mode = ExecutionMode.SHOT_SIMULATION
        cd_res = CDWarmQAOASolver(
            reps_p=2, shots=req.shots, seed=req.seed
        ).solve(ising, hubo=hubo)
        raw_best_bits = cd_res.best_bitstring
        counts = cd_res.bitstring_counts
        for idx, e_val in enumerate(cd_res.energy_trace[::3]):
            iter_telemetry.append(
                {"iteration": idx + 1, "expectation_energy": e_val}
            )
    elif algo == AlgorithmType.MPS:
        exec_mode = ExecutionMode.SHOT_SIMULATION
        mps_res = MPSSimulator(
            max_bond_dim=req.mps_bond_dim, shots=min(req.shots, 512), seed=req.seed
        ).solve(ising)
        raw_best_bits = mps_res.best_bitstring
        counts = mps_res.bitstring_counts
        iter_telemetry.append(
            {
                "iteration": 1,
                "expectation_energy": mps_res.expectation_energy,
                "truncation_error": mps_res.truncation_error,
            }
        )
    elif algo == AlgorithmType.EXACT:
        exec_mode = ExecutionMode.CLASSICAL_REFERENCE
        ex_res = ExactSpectralSolver().solve(ising)
        raw_best_bits = ex_res.best_bitstring
        counts = ex_res.bitstring_counts
        iter_telemetry.append(
            {"iteration": 1, "expectation_energy": ex_res.expectation_energy}
        )
    elif algo == AlgorithmType.STRUCTURED_ENUM:
        exec_mode = ExecutionMode.CLASSICAL_REFERENCE
        se_res = StructuredEnumerationSolver().solve_problem(prob, hubo)
        raw_best_bits = se_res.best_bitstring
        counts = se_res.bitstring_counts or {raw_best_bits: 1}
        iter_telemetry.append(
            {"iteration": 1, "expectation_energy": se_res.expectation_energy}
        )
    elif algo == AlgorithmType.GREEDY:
        exec_mode = ExecutionMode.CLASSICAL_REFERENCE
        gr_res = GreedyOperationalSolver().solve_problem(prob, hubo)
        raw_best_bits = gr_res.best_bitstring
        counts = gr_res.bitstring_counts
        iter_telemetry.append(
            {"iteration": 1, "expectation_energy": gr_res.expectation_energy}
        )
    elif algo == AlgorithmType.SIMULATED_ANNEALING:
        exec_mode = ExecutionMode.CLASSICAL_REFERENCE
        sa_res = SimulatedAnnealingSolver(seed=req.seed).solve(ising)
        raw_best_bits = sa_res.best_bitstring
        counts = sa_res.bitstring_counts
        iter_telemetry.append(
            {"iteration": 1, "expectation_energy": sa_res.expectation_energy}
        )
    else:
        exec_mode = ExecutionMode.SHOT_SIMULATION
        bf_cfg = BFDCQOConfig(max_iterations=req.bf_iterations)
        bf_res = BFDCQOEngine(
            config=bf_cfg,
            shots=req.shots,
            seed=req.seed,
            use_shot_estimates=True,
        ).run(ising)
        raw_best_bits = bf_res.best_bitstring
        counts = bf_res.bitstring_counts
        iter_telemetry = [asdict(it) for it in bf_res.iteration_history]

    # Calibrated Readout Mitigation + CVaR + Real Unitary-Folded ZNE
    job.transition_to(JobState.MITIGATING_ERRORS)
    if req.enable_trex and counts:
        counts = TREXReadoutMitigator().mitigate_counts(
            counts, ising.num_qubits
        )

    cvar_e = (
        compute_cvar_expectation(
            counts, ising.evaluate_bitstring, alpha=req.cvar_alpha
        )
        if counts
        else ising.evaluate_bitstring(raw_best_bits)
    )

    if req.enable_zne and ising.num_qubits <= 10:
        pauli_rep = compile_paulihedral_schedule(ising)
        assert pauli_rep.optimized_ir is not None
        zne_rep = execute_circuit_zne(
            pauli_rep.optimized_ir,
            ising,
            scale_factors=(1, 3, 5),
            per_gate_depol_error=0.008,
            shots=min(req.shots, 1024),
            seed=req.seed,
        )
        zne_e = zne_rep.extrapolated_energy
        zne_se = zne_rep.extrapolated_std_error
    else:
        zne_e = cvar_e
        zne_se = 0.0

    # Independent Validation + Constrained MILP Minimum-Hamming Repair
    job.transition_to(JobState.REPAIRING_FEASIBILITY)
    repair_engine = FeasibilityRepairEngine(prob, hubo)
    decoded = repair_engine.repair_and_decode(raw_best_bits)

    exact_e: float | None = None
    ar: float | None = None
    if ising.num_qubits <= 14:
        spec = analyze_ising_spectrum(ising)
        exact_e = round(spec.ground_energy, 6)
        if decoded.is_feasible:
            ar = round(
                compute_approximation_ratio(
                    decoded.repaired_energy, spec.ground_energy, spec.max_energy
                ),
                5,
            )

    exec_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
    job.transition_to(JobState.COMPLETED)
    GLOBAL_TELEMETRY.record_run(
        solver_name=f"{algo.value}:{req.sdk.value}",
        latency_ms=exec_ms,
        ar=ar if ar is not None else 0.0,
        repaired=decoded.was_repaired,
    )

    sched_dict = asdict(decoded)
    sched_dict["validity"] = decoded.validity.value
    sched_dict["solver_status"] = decoded.solver_status.value
    sched_dict.pop("validation_report", None)
    sched_dict.pop("repair_time_ms", None)

    return OptimizeResponse(
        job_id=job.job_id,
        problem_name=prob.name,
        hamiltonian_sha256=h_sha,
        num_binary_variables=prob.total_binary_variables,
        num_qubits_executed=num_qubits_used,
        estimated_memory_mb=admission.estimated_memory_mb,
        algorithm=algo.value,
        sdk=req.sdk.value,
        execution_mode=exec_mode.value,
        operational_cost_f=decoded.repaired_operational_cost_f,
        best_energy=decoded.repaired_energy,
        exact_ground_energy=exact_e,
        approximation_ratio=ar,
        cvar_expectation_energy=round(cvar_e, 5),
        zne_extrapolated_energy=round(zne_e, 5),
        zne_std_error=round(zne_se, 5),
        execution_time_ms=exec_ms,
        bitstring_counts=dict(list(counts.items())[:10]),
        schedule=OptimizeSchedulePayload(**sched_dict),
        iteration_telemetry=iter_telemetry,
    )
