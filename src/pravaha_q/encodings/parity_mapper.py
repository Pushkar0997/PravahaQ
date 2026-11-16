"""
LHZ Parity Architecture & Syndrome Constraint Mapper
====================================================
Maps interaction terms (Z_i Z_j and Z_i Z_j Z_k) onto single-qubit parity
variables tilde{Z}_p accompanied by closed-loop 3-body and 4-body plaquette
stabilizer constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass


@dataclass(frozen=True)
class ParityMappedHamiltonian:
    """Represents an LHZ / Parity-compiled interaction graph."""

    num_logical_spins: int
    num_parity_qubits: int
    parity_qubit_labels: list[str]
    local_fields: dict[int, float]
    plaquette_constraints: list[tuple[int, ...]]
    plaquette_penalty_strength: float


def compile_parity_mapping(
    ising: IsingSpinGlass, plaquette_strength: float = 5.0
) -> ParityMappedHamiltonian:
    """
    Transform a 3-local IsingSpinGlass into a Parity / LHZ local-field model
    where every 1-body, 2-body, and 3-body term becomes a single physical Z_p
    rotation, regularized by cycle plaquette parity constraints.
    """
    labels: list[str] = []
    local_fields: dict[int, float] = {}
    term_to_p: dict[tuple[int, ...], int] = {}

    p_idx = 0
    for i, val in sorted(ising.h_linear.items()):
        term = (i,)
        term_to_p[term] = p_idx
        labels.append(f"Z_{i}")
        local_fields[p_idx] = float(val)
        p_idx += 1

    for (i, j), val in sorted(ising.j_quadratic.items()):
        term = (i, j)
        term_to_p[term] = p_idx
        labels.append(f"Z_{i}Z_{j}")
        local_fields[p_idx] = float(val)
        p_idx += 1

    for (i, j, k), val in sorted(ising.k_cubic.items()):
        term = (i, j, k)
        term_to_p[term] = p_idx
        labels.append(f"Z_{i}Z_{j}Z_{k}")
        local_fields[p_idx] = float(val)
        p_idx += 1

    # Identify closed triangles (i, j), (j, k), (i, k) -> 3-qubit plaquette
    plaquettes: list[tuple[int, ...]] = []
    n = ising.num_qubits
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (
                    (i, j) in term_to_p
                    and (j, k) in term_to_p
                    and (i, k) in term_to_p
                ):
                    plaquettes.append(
                        (term_to_p[(i, j)], term_to_p[(j, k)], term_to_p[(i, k)])
                    )
                if (
                    (i, j, k) in term_to_p
                    and (i, j) in term_to_p
                    and (k,) in term_to_p
                ):
                    plaquettes.append(
                        (term_to_p[(i, j, k)], term_to_p[(i, j)], term_to_p[(k,)])
                    )

    return ParityMappedHamiltonian(
        num_logical_spins=n,
        num_parity_qubits=p_idx,
        parity_qubit_labels=labels,
        local_fields=local_fields,
        plaquette_constraints=plaquettes,
        plaquette_penalty_strength=plaquette_strength,
    )
