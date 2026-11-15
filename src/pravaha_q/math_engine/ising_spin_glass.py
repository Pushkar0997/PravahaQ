"""
Exact 3-Local Ising Spin-Glass Mapper
=====================================
Maps any 3-Local HUBOPolynomial E_HUBO(x) over x in {0, 1}^N to the exact
diagonal quantum spin-glass Hamiltonian over Pauli-Z eigenvalues z in {+1, -1}^N:

    H_C = C_0 * I + sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j + sum_{i<j<k} K_{ijk} Z_i Z_j Z_k

Satisfies the exact algebraic identity:
    <x| H_C |x> == E_HUBO(x)   for all 2^N computational basis states |x>.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from pravaha_q.math_engine.hubo_builder import HUBOPolynomial


@dataclass
class IsingSpinGlass:
    """
    Canonical 3-Local Ising Spin-Glass Hamiltonian representation.
    Conventions:
      - Computational basis |0> has Z|0> = +1|0> (binary x_i = 0)
      - Computational basis |1> has Z|1> = -1|1> (binary x_i = 1)
      - Affine bijection: x_i = (1 - z_i) / 2  <==>  z_i = 1 - 2 * x_i
    """

    num_qubits: int
    constant_offset: float = 0.0
    h_linear: dict[int, float] = field(default_factory=dict)
    j_quadratic: dict[tuple[int, int], float] = field(default_factory=dict)
    k_cubic: dict[tuple[int, int, int], float] = field(default_factory=dict)

    @property
    def spectral_scale(self) -> float:
        """Return maximum coefficient magnitude Lambda_norm (RFC-0001)."""
        vals: list[float] = []
        vals.extend(abs(v) for v in self.h_linear.values())
        vals.extend(abs(v) for v in self.j_quadratic.values())
        vals.extend(abs(v) for v in self.k_cubic.values())
        return max(vals) if vals else 1.0

    def normalized_coefficients(
        self,
    ) -> tuple[
        dict[int, float],
        dict[tuple[int, int], float],
        dict[tuple[int, int, int], float],
    ]:
        """Return (h, J, K) normalized by spectral_scale so max magnitude <= 1.0."""
        scale = max(self.spectral_scale, 1e-9)
        h_norm = {i: v / scale for i, v in self.h_linear.items()}
        j_norm = {pair: v / scale for pair, v in self.j_quadratic.items()}
        k_norm = {triad: v / scale for triad, v in self.k_cubic.items()}
        return h_norm, j_norm, k_norm

    def evaluate_spin_energy(self, spins: Iterable[int] | np.ndarray) -> float:
        """Evaluate <z| H_C |z> for a spin configuration z in {+1, -1}^N."""
        z = [int(s) for s in spins]
        energy = float(self.constant_offset)
        for i, h_val in self.h_linear.items():
            energy += h_val * z[i]
        for (i, j), j_val in self.j_quadratic.items():
            energy += j_val * z[i] * z[j]
        for (i, j, k), k_val in self.k_cubic.items():
            energy += k_val * z[i] * z[j] * z[k]
        return energy

    def evaluate_bitstring(self, bitstring: Iterable[int] | np.ndarray | str) -> float:
        """Convert binary bitstring x in {0, 1}^N to spins z = 1 - 2x and evaluate H_C."""
        if isinstance(bitstring, str):
            bits = [int(c) for c in bitstring.strip()]
        else:
            bits = [int(b) for b in bitstring]
        spins = [1 - 2 * b for b in bits]
        return self.evaluate_spin_energy(spins)

    def to_coupling_matrix(self) -> np.ndarray:
        """Return symmetric N x N matrix with h_i on diagonal and J_{ij} off-diagonal."""
        mat = np.zeros((self.num_qubits, self.num_qubits), dtype=np.float64)
        for i, h_val in self.h_linear.items():
            mat[i, i] = h_val
        for (i, j), j_val in self.j_quadratic.items():
            mat[i, j] = j_val
            mat[j, i] = j_val
        return mat

    def to_pauli_list(self, include_identity: bool = False) -> list[tuple[str, float]]:
        """
        Export Hamiltonian as a list of (PauliString, coefficient) tuples.
        Note: Uses standard left-to-right qubit indexing (index 0 is leftmost character).
        """
        n = self.num_qubits
        terms: list[tuple[str, float]] = []
        if include_identity and abs(self.constant_offset) > 1e-12:
            terms.append(("I" * n, float(self.constant_offset)))

        for i in sorted(self.h_linear):
            val = self.h_linear[i]
            if abs(val) > 1e-12:
                chars = ["I"] * n
                chars[i] = "Z"
                terms.append(("".join(chars), float(val)))

        for i, j in sorted(self.j_quadratic):
            val = self.j_quadratic[(i, j)]
            if abs(val) > 1e-12:
                chars = ["I"] * n
                chars[i] = "Z"
                chars[j] = "Z"
                terms.append(("".join(chars), float(val)))

        for i, j, k in sorted(self.k_cubic):
            val = self.k_cubic[(i, j, k)]
            if abs(val) > 1e-12:
                chars = ["I"] * n
                chars[i] = "Z"
                chars[j] = "Z"
                chars[k] = "Z"
                terms.append(("".join(chars), float(val)))

        return terms

    def diagonal_vector(self) -> np.ndarray:
        """
        Compute the exact 2^N diagonal eigenvalue vector of H_C in the computational
        basis (where integer index idx in [0, 2^N - 1] corresponds to big-endian bitstring).
        Fast vectorized implementation using bitwise operations.
        """
        n = self.num_qubits
        num_states = 1 << n
        indices = np.arange(num_states, dtype=np.uint32)

        # z_i = +1 if bit i is 0, else -1
        spins = np.empty((n, num_states), dtype=np.float64)
        for q in range(n):
            shift = n - 1 - q
            bit = (indices >> shift) & 1
            spins[q] = 1.0 - 2.0 * bit

        diag = np.full(num_states, self.constant_offset, dtype=np.float64)
        for i, h_val in self.h_linear.items():
            diag += h_val * spins[i]
        for (i, j), j_val in self.j_quadratic.items():
            diag += j_val * spins[i] * spins[j]
        for (i, j, k), k_val in self.k_cubic.items():
            diag += k_val * spins[i] * spins[j] * spins[k]

        return diag


def hubo_to_ising(hubo: HUBOPolynomial) -> IsingSpinGlass:
    """
    Transform a 3-Local HUBOPolynomial E_HUBO(x) into its exact IsingSpinGlass H_C
    via x_i = (1 - Z_i) / 2 (Theorem 1 of docs/03_MATHEMATICAL_SPECIFICATION.md).
    """
    c0 = float(hubo.constant_offset)
    h: dict[int, float] = {}
    j_map: dict[tuple[int, int], float] = {}
    k_map: dict[tuple[int, int, int], float] = {}

    # 1. Transform 1-body linear terms: L_i * (1 - z_i) / 2
    for i, l_val in hubo.linear.items():
        c0 += 0.5 * l_val
        h[i] = h.get(i, 0.0) - 0.5 * l_val

    # 2. Transform 2-body quadratic terms: Q_{ij} * (1 - z_i)(1 - z_j) / 4
    for (i, j), q_val in hubo.quadratic.items():
        u, v = (min(i, j), max(i, j))
        c0 += 0.25 * q_val
        h[u] = h.get(u, 0.0) - 0.25 * q_val
        h[v] = h.get(v, 0.0) - 0.25 * q_val
        j_map[(u, v)] = j_map.get((u, v), 0.0) + 0.25 * q_val

    # 3. Transform 3-body cubic terms: T_{ijk} * (1 - z_i)(1 - z_j)(1 - z_k) / 8
    for (i, j, k), t_val in hubo.cubic.items():
        u, v, w = sorted((i, j, k))
        w8 = 0.125 * t_val
        c0 += w8
        h[u] = h.get(u, 0.0) - w8
        h[v] = h.get(v, 0.0) - w8
        h[w] = h.get(w, 0.0) - w8

        j_map[(u, v)] = j_map.get((u, v), 0.0) + w8
        j_map[(u, w)] = j_map.get((u, w), 0.0) + w8
        j_map[(v, w)] = j_map.get((v, w), 0.0) + w8

        k_map[(u, v, w)] = k_map.get((u, v, w), 0.0) - w8

    # Prune near-zero numerical noise
    h_clean = {idx: val for idx, val in h.items() if abs(val) > 1e-12}
    j_clean = {pair: val for pair, val in j_map.items() if abs(val) > 1e-12}
    k_clean = {triad: val for triad, val in k_map.items() if abs(val) > 1e-12}

    return IsingSpinGlass(
        num_qubits=hubo.num_variables,
        constant_offset=c0,
        h_linear=h_clean,
        j_quadratic=j_clean,
        k_cubic=k_clean,
    )
