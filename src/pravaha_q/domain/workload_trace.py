"""Distributed LLM / AI inference batch workload and KV-cache coupling models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class InferenceBatchTask:
    """Represents an incoming AI inference batch (e.g., LLM prefill or decode stage)."""

    task_id: int
    name: str
    compute_flops_giga: float
    thermal_load_watts: float
    priority_weight: float = 1.0
    deadline_ms: float = 250.0


@dataclass(frozen=True)
class WorkloadGraph:
    """Collection of inference tasks and their pairwise KV-cache synchronization weights."""

    tasks: tuple[InferenceBatchTask, ...]
    kv_sync_weights: tuple[tuple[float, ...], ...] = field(default_factory=tuple)

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    def sync_coupling(self, task_m: int, task_mp: int) -> float:
        """Return the KV-cache communication coupling weight between two tasks."""
        if task_m == task_mp or not self.kv_sync_weights:
            return 0.0
        return float(self.kv_sync_weights[task_m][task_mp])

    def to_sync_matrix(self) -> np.ndarray:
        """Return the M x M symmetric KV-cache synchronization matrix."""
        m = self.num_tasks
        mat = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                mat[i, j] = self.sync_coupling(i, j)
        return mat
