"""
Backtracking Iterative Quantum Branch-and-Bound (IQO-BnB)
=========================================================
Uses the exact connected two-point spin covariance:
    Cov(Z_i, Z_j) = <Z_i Z_j> - <Z_i><Z_j>
which is identically zero for any unentangled product state |psi> = bigotimes_q |phi_q>.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pravaha_q.algorithms.bf_dcqo import BFDCQOEngine
from pravaha_q.config import BFDCQOConfig
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass


@dataclass(frozen=True)
class IQOReductionStep:
    """Records a single spin-pair variable elimination step in the IQO-BnB tree."""

    remaining_qubits: int
    eliminated_qubit: int
    reference_qubit: int
    parity_sign: int
    correlation_magnitude: float
    is_backtrack_branch: bool


@dataclass(frozen=True)
class IQOBranchAndBoundResult:
    """Telemetry and global solution found by Backtracking IQO-BnB."""

    num_qubits: int
    branches_explored: int
    greedy_rqaoa_energy: float
    backtrack_improved_over_greedy: bool
    best_bitstring: str
    best_energy: float
    reduction_trace: list[IQOReductionStep] = field(default_factory=list)


def compute_connected_covariance_matrix(
    probs: np.ndarray, num_qubits: int
) -> np.ndarray:
    """
    Compute the exact N x N connected spin covariance matrix:
        Cov(Z_i, Z_j) = <Z_i Z_j> - <Z_i><Z_j>
    For any product state |psi> = bigotimes_q |phi_q>, all off-diagonal entries
    Cov(Z_i, Z_j) (i != j) are identically 0.0.
    """
    n = num_qubits
    dim = 1 << n
    indices = np.arange(dim, dtype=np.uint32)
    spins = np.empty((n, dim), dtype=np.float64)
    for q in range(n):
        bit = (indices >> (n - 1 - q)) & 1
        spins[q] = 1.0 - 2.0 * bit

    mags = np.sum(probs * spins, axis=1)
    cov = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        cov[i, i] = 1.0 - float(mags[i] ** 2)
        for j in range(i + 1, n):
            zz_exp = float(np.sum(probs * spins[i] * spins[j]))
            c_ij = zz_exp - float(mags[i] * mags[j])
            cov[i, j] = c_ij
            cov[j, i] = c_ij
    return cov


def compute_strongest_spin_correlator(
    probs: np.ndarray, ising: IsingSpinGlass
) -> tuple[int, int, float]:
    """
    Find qubit pair (i, j) with i < j maximizing |Cov(Z_i, Z_j)| = |<Z_i Z_j> - <Z_i><Z_j>|.
    """
    n = ising.num_qubits
    cov = compute_connected_covariance_matrix(probs, n)

    best_pair = (0, 1)
    best_score = -1.0
    best_cov = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            c_ij = float(cov[i, j])
            has_edge = 1.15 if (i, j) in ising.j_quadratic else 1.0
            score = abs(c_ij) * has_edge
            if score > best_score:
                best_score = score
                best_pair = (i, j)
                best_cov = c_ij

    return best_pair[0], best_pair[1], best_cov


class IQOBranchAndBoundSolver:
    """Executes Backtracking Iterative Quantum Optimization (IQO-BnB)."""

    def __init__(
        self,
        classical_cutoff_vars: int = 4,
        max_backtrack_depth: int = 2,
        seed: int = 42,
    ) -> None:
        self.classical_cutoff_vars = max(classical_cutoff_vars, 2)
        self.max_backtrack_depth = max_backtrack_depth
        self.seed = seed
        self._bf_engine = BFDCQOEngine(
            config=BFDCQOConfig(max_iterations=2, impulse_dt=0.42),
            shots=512,
            seed=seed,
        )

    def solve(self, ising: IsingSpinGlass) -> IQOBranchAndBoundResult:
        active_qubits = list(range(ising.num_qubits))
        branches_explored = 0
        best_energy = float("inf")
        best_spins_map: dict[int, int] = {}
        best_trace: list[IQOReductionStep] = []
        greedy_energy: float | None = None

        def explore(
            sub_ising: IsingSpinGlass,
            qubit_map: list[int],
            substitutions: list[tuple[int, int, int]],
            trace: list[IQOReductionStep],
            backtracks_used: int,
            is_greedy_path: bool,
        ) -> None:
            nonlocal branches_explored, best_energy, best_spins_map, best_trace, greedy_energy

            n_curr = sub_ising.num_qubits
            if n_curr <= self.classical_cutoff_vars:
                branches_explored += 1
                diag = sub_ising.diagonal_vector()
                min_idx = int(np.argmin(diag))
                core_bits = format(min_idx, f"0{n_curr}b")
                core_spins = {
                    qubit_map[idx]: (1 - 2 * int(b))
                    for idx, b in enumerate(core_bits)
                }
                for elim_q, ref_q, sigma in reversed(substitutions):
                    core_spins[elim_q] = int(sigma * core_spins[ref_q])

                full_spins = [core_spins[q] for q in range(ising.num_qubits)]
                full_e = ising.evaluate_spin_energy(full_spins)

                if is_greedy_path and greedy_energy is None:
                    greedy_energy = full_e

                if full_e < best_energy:
                    best_energy = full_e
                    best_spins_map = core_spins
                    best_trace = list(trace)
                return

            bf_res = self._bf_engine.run(sub_ising)
            state_probs = self._reconstruct_prob_vector(
                bf_res.statevector_probabilities, n_curr
            )
            i_local, j_local, cov_val = compute_strongest_spin_correlator(
                state_probs, sub_ising
            )
            primary_sigma = 1 if cov_val >= 0.0 else -1

            orig_i = qubit_map[i_local]
            orig_j = qubit_map[j_local]

            red_primary, new_map = reduce_ising_spin_pair(
                sub_ising, qubit_map, i_local, j_local, primary_sigma
            )
            step_prim = IQOReductionStep(
                remaining_qubits=n_curr - 1,
                eliminated_qubit=orig_j,
                reference_qubit=orig_i,
                parity_sign=primary_sigma,
                correlation_magnitude=round(abs(cov_val), 5),
                is_backtrack_branch=False,
            )
            explore(
                red_primary,
                new_map,
                substitutions + [(orig_j, orig_i, primary_sigma)],
                trace + [step_prim],
                backtracks_used,
                is_greedy_path,
            )

            if backtracks_used < self.max_backtrack_depth:
                opp_sigma = -primary_sigma
                red_bt, new_map_bt = reduce_ising_spin_pair(
                    sub_ising, qubit_map, i_local, j_local, opp_sigma
                )
                step_bt = IQOReductionStep(
                    remaining_qubits=n_curr - 1,
                    eliminated_qubit=orig_j,
                    reference_qubit=orig_i,
                    parity_sign=opp_sigma,
                    correlation_magnitude=round(abs(cov_val), 5),
                    is_backtrack_branch=True,
                )
                explore(
                    red_bt,
                    new_map_bt,
                    substitutions + [(orig_j, orig_i, opp_sigma)],
                    trace + [step_bt],
                    backtracks_used + 1,
                    False,
                )

        explore(
            sub_ising=ising,
            qubit_map=active_qubits,
            substitutions=[],
            trace=[],
            backtracks_used=0,
            is_greedy_path=True,
        )

        best_bits = [
            (1 - best_spins_map.get(q, 1)) // 2 for q in range(ising.num_qubits)
        ]
        best_bitstr = "".join(str(b) for b in best_bits)
        g_e = greedy_energy if greedy_energy is not None else best_energy

        return IQOBranchAndBoundResult(
            num_qubits=ising.num_qubits,
            branches_explored=branches_explored,
            greedy_rqaoa_energy=round(float(g_e), 6),
            backtrack_improved_over_greedy=bool(best_energy < g_e - 1e-6),
            best_bitstring=best_bitstr,
            best_energy=round(float(best_energy), 6),
            reduction_trace=best_trace,
        )

    @staticmethod
    def _reconstruct_prob_vector(
        sparse_probs: dict[str, float], num_qubits: int
    ) -> np.ndarray:
        vec = np.full(1 << num_qubits, 1e-7, dtype=np.float64)
        for bitstr, p in sparse_probs.items():
            vec[int(bitstr, 2)] = float(p)
        return vec / np.sum(vec)


def reduce_ising_spin_pair(
    ising: IsingSpinGlass,
    qubit_map: list[int],
    i_keep: int,
    j_elim: int,
    sigma: int,
) -> tuple[IsingSpinGlass, list[int]]:
    """Substitute Z_{j_elim} = sigma * Z_{i_keep} across 1-body, 2-body, and 3-body terms."""
    n = ising.num_qubits
    new_map = [qubit_map[q] for q in range(n) if q != j_elim]

    def remap(q: int) -> int:
        actual = i_keep if q == j_elim else q
        return actual if actual < j_elim else actual - 1

    c0 = float(ising.constant_offset)
    h_new: dict[int, float] = {}
    j_new: dict[tuple[int, int], float] = {}
    k_new: dict[tuple[int, int, int], float] = {}

    for q, val in ising.h_linear.items():
        mult = sigma if q == j_elim else 1
        rq = remap(q)
        h_new[rq] = h_new.get(rq, 0.0) + mult * val

    for (u, v), val in ising.j_quadratic.items():
        if (u == i_keep and v == j_elim) or (u == j_elim and v == i_keep):
            c0 += sigma * val
        else:
            mult = sigma if (u == j_elim or v == j_elim) else 1
            ru, rv = remap(u), remap(v)
            pair = (min(ru, rv), max(ru, rv))
            j_new[pair] = j_new.get(pair, 0.0) + mult * val

    for (u, v, w), val in ising.k_cubic.items():
        triad_set = {u, v, w}
        if i_keep in triad_set and j_elim in triad_set:
            other = next(x for x in (u, v, w) if x not in (i_keep, j_elim))
            ro = remap(other)
            h_new[ro] = h_new.get(ro, 0.0) + sigma * val
        else:
            mult = sigma if j_elim in triad_set else 1
            ru, rv, rw = sorted((remap(u), remap(v), remap(w)))
            k_new[(ru, rv, rw)] = k_new.get((ru, rv, rw), 0.0) + mult * val

    return (
        IsingSpinGlass(
            num_qubits=n - 1,
            constant_offset=c0,
            h_linear={k: v for k, v in h_new.items() if abs(v) > 1e-12},
            j_quadratic={k: v for k, v in j_new.items() if abs(v) > 1e-12},
            k_cubic={k: v for k, v in k_new.items() if abs(v) > 1e-12},
        ),
        new_map,
    )
