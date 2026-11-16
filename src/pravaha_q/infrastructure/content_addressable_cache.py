"""SHA-256 Content-Addressable Circuit & Hamiltonian Caching."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass


def compute_hamiltonian_sha256(ising: IsingSpinGlass) -> str:
    """Compute a deterministic SHA-256 fingerprint for an Ising Hamiltonian."""
    payload: dict[str, Any] = {
        "n": ising.num_qubits,
        "c0": round(ising.constant_offset, 6),
        "h": {str(k): round(v, 6) for k, v in sorted(ising.h_linear.items())},
        "j": {str(k): round(v, 6) for k, v in sorted(ising.j_quadratic.items())},
        "k": {str(k): round(v, 6) for k, v in sorted(ising.k_cubic.items())},
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
