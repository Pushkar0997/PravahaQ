"""Prometheus Telemetry & Structured Audit Logging for PravahaQ."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TelemetryCollector:
    """In-memory telemetry aggregator exportable in Prometheus text format."""

    total_jobs_executed: int = 0
    cache_hits: int = 0
    total_feasibility_repairs: int = 0
    last_approximation_ratio: float = 1.0
    solver_latency_ms: dict[str, float] = field(default_factory=dict)

    def record_run(
        self, solver_name: str, latency_ms: float, ar: float, repaired: bool
    ) -> None:
        self.total_jobs_executed += 1
        self.last_approximation_ratio = float(ar)
        self.solver_latency_ms[solver_name] = round(float(latency_ms), 2)
        if repaired:
            self.total_feasibility_repairs += 1

    def to_prometheus_text(self) -> str:
        lines = [
            "# HELP pravaha_q_jobs_total Total hybrid quantum jobs executed",
            "# TYPE pravaha_q_jobs_total counter",
            f"pravaha_q_jobs_total {self.total_jobs_executed}",
            "# HELP pravaha_q_last_approximation_ratio Latest Approximation Ratio",
            "# TYPE pravaha_q_last_approximation_ratio gauge",
            f"pravaha_q_last_approximation_ratio {self.last_approximation_ratio:.4f}",
            "# HELP pravaha_q_feasibility_repairs_total Total post-quantum schedule repairs",
            "# TYPE pravaha_q_feasibility_repairs_total counter",
            f"pravaha_q_feasibility_repairs_total {self.total_feasibility_repairs}",
        ]
        for s_name, lat in self.solver_latency_ms.items():
            lines.append(
                f'pravaha_q_solver_latency_ms{{solver="{s_name}"}} {lat:.2f}'
            )
        return "\n".join(lines) + "\n"


GLOBAL_TELEMETRY = TelemetryCollector()
