"""Scientific benchmarking, 2D landscape tomography, ablation & shadow-mode evaluators."""

from pravaha_q.benchmarks.landscape_tomography import (
    LandscapeSurface2D,
    scan_energy_landscape_2d,
)
from pravaha_q.benchmarks.metrics import compute_approximation_ratio
from pravaha_q.benchmarks.shadow_evaluator import (
    ShadowControllerMetrics,
    generate_recorded_workload_trace,
    run_shadow_mode_evaluation,
)
from pravaha_q.benchmarks.suite_runner import (
    AggregatedMethodStats,
    BenchmarkRecord,
    aggregate_method_statistics,
    run_full_ablation_suite,
    solve_separate_routing_then_mode,
)

__all__ = [
    "AggregatedMethodStats",
    "BenchmarkRecord",
    "LandscapeSurface2D",
    "ShadowControllerMetrics",
    "aggregate_method_statistics",
    "compute_approximation_ratio",
    "generate_recorded_workload_trace",
    "run_full_ablation_suite",
    "run_shadow_mode_evaluation",
    "scan_energy_landscape_2d",
    "solve_separate_routing_then_mode",
]
