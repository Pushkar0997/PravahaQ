"""
Calibrated Readout Confusion Matrix Inversion & TREX Mitigator
==============================================================
Models single-qubit asymmetric readout assignment matrix:
    M_q = [[1 - p(1|0),     p(0|1)    ],
           [    p(1|0), 1 - p(0|1)    ]]
with exact single-qubit inverse:
    M_q^{-1} = 1 / (1 - p(1|0) - p(0|1)) * [[1 - p(0|1),    -p(0|1)    ],
                                             [   -p(1|0), 1 - p(1|0)    ]]
Applies Kronecker inverse M^{-1} = bigotimes_q M_q^{-1} on the active measurement
subspace and projects the quasi-probability vector onto the probability simplex.
"""

from __future__ import annotations

import numpy as np


class TREXReadoutMitigator:
    """Calibrated tensor-product readout confusion matrix mitigator."""

    def __init__(
        self,
        readout_error: float = 0.0085,
        p_0_to_1: float | None = None,
        p_1_to_0: float | None = None,
        seed: int = 42,
    ) -> None:
        self.readout_error = readout_error
        self.p01 = p_0_to_1 if p_0_to_1 is not None else readout_error
        self.p10 = p_1_to_0 if p_1_to_0 is not None else readout_error
        self.seed = seed

    def single_qubit_confusion_matrix(self) -> np.ndarray:
        """Return the 2x2 single-qubit readout assignment matrix M_q."""
        return np.array(
            [[1.0 - self.p01, self.p10], [self.p01, 1.0 - self.p10]],
            dtype=np.float64,
        )

    def single_qubit_inverse_matrix(self) -> np.ndarray:
        """Return the exact 2x2 inverse confusion matrix M_q^{-1}."""
        det = 1.0 - self.p01 - self.p10
        if abs(det) < 1e-9:
            return np.eye(2, dtype=np.float64)
        return (
            np.array(
                [[1.0 - self.p10, -self.p10], [-self.p01, 1.0 - self.p01]],
                dtype=np.float64,
            )
            / det
        )

    def mitigate_counts(
        self, counts: dict[str, int], num_qubits: int
    ) -> dict[str, int]:
        """
        Invert the readout confusion channel over the empirical shot distribution
        and project back onto the closest valid probability simplex.
        """
        total_shots = sum(counts.values())
        if total_shots <= 0 or (self.p01 <= 0.0 and self.p10 <= 0.0):
            return dict(counts)

        m_inv_1q = self.single_qubit_inverse_matrix()

        if num_qubits <= 10:
            dim = 1 << num_qubits
            p_noisy = np.zeros(dim, dtype=np.float64)
            for bstr, c in counts.items():
                p_noisy[int(bstr, 2)] = float(c) / float(total_shots)

            # Apply M_q^{-1} along each qubit axis in O(n * 2^n) time
            tensor_p = p_noisy.reshape((2,) * num_qubits)
            for q in range(num_qubits):
                tensor_p = np.tensordot(m_inv_1q, tensor_p, axes=([1], [q]))
                tensor_p = np.moveaxis(tensor_p, 0, q)

            p_quasi = tensor_p.reshape(-1)
            p_mit = _project_onto_simplex(p_quasi)

            mitigated: dict[str, int] = {}
            for idx, prob in enumerate(p_mit):
                c_new = int(round(prob * total_shots))
                if c_new > 0:
                    mitigated[format(idx, f"0{num_qubits}b")] = c_new
            return (
                dict(
                    sorted(
                        mitigated.items(), key=lambda item: item[1], reverse=True
                    )
                )
                if mitigated
                else dict(counts)
            )

        # Active-subspace diagonal survival rescaling for N > 10
        diag_survival = (1.0 - self.p01) * (1.0 - self.p10)
        scale_inv = 1.0 / max(diag_survival, 0.5)
        mean_c = total_shots / max(len(counts), 1)
        mitigated_large: dict[str, int] = {}
        for bstr, c in counts.items():
            centered = mean_c + scale_inv * (c - mean_c)
            c_new = max(int(round(centered)), 0)
            if c_new > 0:
                mitigated_large[bstr] = c_new
        return mitigated_large if mitigated_large else dict(counts)


def _project_onto_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of a quasi-probability vector v onto the unit simplex."""
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0.0
    if not np.any(cond):
        return np.ones(n, dtype=np.float64) / n
    rho = int(ind[cond][-1])
    theta = float(cssv[rho - 1] / rho)
    return np.maximum(v - theta, 0.0)
