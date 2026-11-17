"""
Classical Reference Implementations (Step 3)
============================================
Establishes ground-truth operational solutions before evaluating quantum heuristics:
  1. `StructuredEnumerationSolver`: Exhaustively enumerates the structured one-hot
     decision space of size N_node^M * R^{N_node} (e.g., 4^8 = 65,536 instead of 2^36),
     validates every schedule via `ScheduleValidator`, and finds the exact minimum
     of the true operational objective F(x*) (or proves `PROVEN_INFEASIBLE`).
  2. `GreedyOperationalSolver`: Fast O(M * N_node) operational baseline.
  3. `SimulatedAnnealingSolver`: Classical thermal annealing baseline over E(x).
  4. `ExactSpectralSolver`: Exact 2^N diagonalization of the unconstrained Ising H_C.
"""

from __future__ import annotations

from itertools import product
import time

import numpy as np

from pravaha_q.domain.problem_generator import CoupledInfrastructureProblem
from pravaha_q.math_engine.hubo_builder import HUBOPolynomial
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass
from pravaha_q.math_engine.validator import (
    ScheduleValidator,
    ScheduleValidity,
    SolverStatus,
)
from pravaha_q.solvers.base import ExecutionMode, SolverResult


class StructuredEnumerationSolver:
    """
    Exact classical reference solver that enumerates the structured one-hot
    assignment space N_node^M * R^{N_node} and evaluates F(x) on validated schedules.
    """

    def __init__(self, max_combinations: int = 200_000) -> None:
        self.max_combinations = max_combinations

    def solve_problem(
        self,
        problem: CoupledInfrastructureProblem,
        hubo: HUBOPolynomial | None = None,
    ) -> SolverResult:
        t0 = time.perf_counter()
        M = problem.num_tasks
        N_node = problem.num_nodes
        R = problem.num_power_states if problem.has_mode_variables else 1
        total_combos = (N_node**M) * (R**N_node)

        if total_combos > self.max_combinations:
            raise ValueError(
                f"Structured space {total_combos} exceeds max_combinations={self.max_combinations}."
            )

        validator = ScheduleValidator(problem)
        N_vars = problem.total_binary_variables
        best_f = float("inf")
        best_bits: list[int] | None = None
        feasible_count = 0

        mode_iter = (
            product(range(R), repeat=N_node)
            if problem.has_mode_variables
            else [tuple(0 for _ in range(N_node))]
        )

        for mode_tuple in mode_iter:
            caps = [problem.node_mode_capacity(n, mode_tuple[n]) for n in range(N_node)]
            for task_tuple in product(range(N_node), repeat=M):
                loads = [0.0] * N_node
                overloaded = False
                for m, n_assigned in enumerate(task_tuple):
                    loads[n_assigned] += problem.workload.tasks[m].thermal_load_watts
                    if loads[n_assigned] > caps[n_assigned] + 1e-6:
                        overloaded = True
                        break
                if overloaded:
                    continue

                x = [0] * N_vars
                for m, n_assigned in enumerate(task_tuple):
                    x[problem.task_node_var_index(m, n_assigned)] = 1
                if problem.has_mode_variables:
                    for n, r_assigned in enumerate(mode_tuple):
                        x[problem.node_power_var_index(n, r_assigned)] = 1

                rep = validator.validate(x)
                if rep.is_valid and rep.operational_cost_f is not None:
                    feasible_count += 1
                    if rep.operational_cost_f < best_f:
                        best_f = rep.operational_cost_f
                        best_bits = x

        exec_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if best_bits is None:
            fallback_str = "0" * N_vars
            return SolverResult(
                solver_name="StructuredEnumeration",
                sdk_name="Classical-Exact",
                num_qubits=N_vars,
                best_bitstring=fallback_str,
                best_energy=float("inf"),
                expectation_energy=float("inf"),
                approximation_ratio=None,
                execution_time_ms=exec_ms,
                execution_mode=ExecutionMode.CLASSICAL_REFERENCE,
                operational_cost_f=None,
                validity=ScheduleValidity.INVALID,
                solver_status=SolverStatus.PROVEN_INFEASIBLE,
                metadata={"enumerated_states": total_combos, "feasible_states": 0},
            )

        best_str = "".join(str(b) for b in best_bits)
        e_surrogate = hubo.evaluate(best_bits) if hubo is not None else best_f
        return SolverResult(
            solver_name="StructuredEnumeration",
            sdk_name="Classical-Exact",
            num_qubits=N_vars,
            best_bitstring=best_str,
            best_energy=round(e_surrogate, 6),
            expectation_energy=round(e_surrogate, 6),
            approximation_ratio=1.0,
            execution_time_ms=exec_ms,
            bitstring_counts={best_str: 1},
            execution_mode=ExecutionMode.CLASSICAL_REFERENCE,
            operational_cost_f=round(best_f, 6),
            validity=ScheduleValidity.VALID,
            solver_status=SolverStatus.OPTIMAL,
            metadata={"enumerated_states": total_combos, "feasible_states": feasible_count},
        )


class GreedyOperationalSolver:
    """Fast deterministic O(M * N_node) greedy operational baseline."""

    def solve_problem(
        self,
        problem: CoupledInfrastructureProblem,
        hubo: HUBOPolynomial | None = None,
    ) -> SolverResult:
        t0 = time.perf_counter()
        M = problem.num_tasks
        N_node = problem.num_nodes
        R = problem.num_power_states
        N_vars = problem.total_binary_variables

        chosen_modes = [0] * N_node
        loads = [0.0] * N_node
        chosen_nodes = [0] * M

        task_order = sorted(
            range(M),
            key=lambda m: problem.workload.tasks[m].thermal_load_watts,
            reverse=True,
        )
        for m in task_order:
            q_m = problem.workload.tasks[m].thermal_load_watts
            best_n = 0
            best_score = float("inf")
            for n in range(N_node):
                max_cap_n = max(
                    problem.node_mode_capacity(n, r) for r in range(R)
                )
                if loads[n] + q_m <= max_cap_n + 1e-6:
                    score = problem.task_execution_latency_ms(m, n)
                    if score < best_score:
                        best_score = score
                        best_n = n
            chosen_nodes[m] = best_n
            loads[best_n] += q_m

        if problem.has_mode_variables:
            for n in range(N_node):
                valid_modes = [
                    r for r in range(R) if problem.node_mode_capacity(n, r) >= loads[n] - 1e-6
                ]
                if valid_modes:
                    chosen_modes[n] = min(
                        valid_modes,
                        key=lambda r: problem.microgrid.power_states[r].energy_cost_tariff,
                    )
                else:
                    chosen_modes[n] = max(
                        range(R), key=lambda r: problem.node_mode_capacity(n, r)
                    )

        x = [0] * N_vars
        for m, n_assigned in enumerate(chosen_nodes):
            x[problem.task_node_var_index(m, n_assigned)] = 1
        if problem.has_mode_variables:
            for n, r_assigned in enumerate(chosen_modes):
                x[problem.node_power_var_index(n, r_assigned)] = 1

        rep = ScheduleValidator(problem).validate(x)
        bstr = "".join(str(b) for b in x)
        e_val = hubo.evaluate(x) if hubo is not None else (rep.operational_cost_f or 0.0)
        exec_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return SolverResult(
            solver_name="GreedyOperational",
            sdk_name="Classical-Heuristic",
            num_qubits=N_vars,
            best_bitstring=bstr,
            best_energy=round(e_val, 6),
            expectation_energy=round(e_val, 6),
            approximation_ratio=None,
            execution_time_ms=exec_ms,
            bitstring_counts={bstr: 1},
            execution_mode=ExecutionMode.CLASSICAL_REFERENCE,
            operational_cost_f=rep.operational_cost_f,
            validity=rep.validity,
            solver_status=(
                SolverStatus.FEASIBLE_INCUMBENT if rep.is_valid else SolverStatus.UNKNOWN
            ),
        )


class SimulatedAnnealingSolver:
    """Classical Metropolis-Hastings Simulated Annealing baseline over Ising H_C."""

    def __init__(self, steps: int = 400, seed: int = 42) -> None:
        self.steps = steps
        self.seed = seed

    def solve(self, ising: IsingSpinGlass) -> SolverResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        n = ising.num_qubits
        spins = rng.choice([-1, 1], size=n)
        curr_e = ising.evaluate_spin_energy(spins)
        best_e = curr_e
        best_spins = spins.copy()

        temps = np.geomspace(5.0, 0.08, self.steps)
        for temp in temps:
            flip_idx = int(rng.integers(0, n))
            spins[flip_idx] *= -1
            cand_e = ising.evaluate_spin_energy(spins)
            delta = cand_e - curr_e
            if delta <= 0.0 or rng.random() < np.exp(-delta / temp):
                curr_e = cand_e
                if curr_e < best_e:
                    best_e = curr_e
                    best_spins = spins.copy()
            else:
                spins[flip_idx] *= -1

        best_bits = [(1 - int(s)) // 2 for s in best_spins]
        bstr = "".join(str(b) for b in best_bits)
        exec_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return SolverResult(
            solver_name="SimulatedAnnealing",
            sdk_name="Classical-Annealer",
            num_qubits=n,
            best_bitstring=bstr,
            best_energy=round(best_e, 6),
            expectation_energy=round(curr_e, 6),
            approximation_ratio=None,
            execution_time_ms=exec_ms,
            bitstring_counts={bstr: 1},
            execution_mode=ExecutionMode.CLASSICAL_REFERENCE,
            solver_status=SolverStatus.UNKNOWN,
        )


class ExactSpectralSolver:
    """Exact diagonal solver via 2^N statevector enumeration of Ising H_C."""

    def __init__(self) -> None:
        self.name = "ExactSpectral"

    def solve(self, ising: IsingSpinGlass) -> SolverResult:
        start_t = time.perf_counter()
        n = ising.num_qubits
        diag = ising.diagonal_vector()
        min_idx = int(np.argmin(diag))
        e_min = float(diag[min_idx])
        best_bstr = format(min_idx, f"0{n}b")
        exec_ms = (time.perf_counter() - start_t) * 1000.0

        return SolverResult(
            solver_name="ExactSpectral",
            sdk_name="NumPy/SciPy",
            num_qubits=n,
            best_bitstring=best_bstr,
            best_energy=round(e_min, 6),
            expectation_energy=round(e_min, 6),
            approximation_ratio=1.0,
            execution_time_ms=round(exec_ms, 2),
            bitstring_counts={best_bstr: 1},
            execution_mode=ExecutionMode.CLASSICAL_REFERENCE,
            solver_status=SolverStatus.OPTIMAL,
        )
