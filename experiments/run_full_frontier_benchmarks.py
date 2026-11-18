"""
PravahaQ (v0.1.0) — Reproducible Honest Ablation & Shadow-Mode Benchmark Runner
===============================================================================
Recomputes all multi-seed ablation statistics (`Random+Repair`, `Separate+Repair`,
`Greedy+Repair`, `SimulatedAnnealing+Repair`, `BF-DCQO+Repair`, `PCE+Repair`,
`StructuredExact`, plus overloaded rejection and Shadow-Mode stream replay)
and writes:
  - `experiments/results/frontier_benchmark_results.json`
  - `experiments/results/paper_tables.tex`
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from pravaha_q.benchmarks.shadow_evaluator import run_shadow_mode_evaluation
from pravaha_q.benchmarks.suite_runner import run_full_ablation_suite

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_benchmark_matrix() -> dict[str, Any]:
    """Execute honest ablation suite and shadow-mode controller evaluation."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "frontier_benchmark_results.json"

    ablation_payload = run_full_ablation_suite(
        output_json=None, seeds=(42, 108, 256)
    )
    shadow_metrics = run_shadow_mode_evaluation(num_slices=5, base_seed=500)
    ablation_payload["shadow_mode_evaluation"] = [
        asdict(m) for m in shadow_metrics
    ]

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(ablation_payload, f, indent=2)

    tex_path = RESULTS_DIR / "paper_tables.tex"
    tex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Recomputed Multi-Seed Operational Cost $F(x)$ and Feasibility Ablation Benchmarks (PravahaQ \\texttt{v0.1.0}).}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "\\textbf{Method / Pipeline} & \\textbf{Execution Mode} & \\textbf{Raw Valid} & \\textbf{Repaired Valid} & \\textbf{Mean $F(x)$} & \\textbf{Worst $F(x)$} & \\textbf{Avg Hamming} & \\textbf{p95 Latency} \\\\",
        "\\midrule",
    ]
    for stat in ablation_payload["aggregated_statistics"]:
        mean_f = (
            f"{stat['mean_operational_cost_f']:.3f}"
            if stat["mean_operational_cost_f"] is not None
            else "N/A"
        )
        worst_f = (
            f"{stat['worst_case_operational_cost_f']:.3f}"
            if stat["worst_case_operational_cost_f"] is not None
            else "N/A"
        )
        tex_lines.append(
            f"{stat['solver_label']} & {stat['execution_mode']} & "
            f"{stat['raw_feasibility_rate_pct']:.1f}\\% & {stat['repaired_feasibility_rate_pct']:.1f}\\% & "
            f"{mean_f} & {worst_f} & {stat['mean_hamming_repair_distance']:.2f} & "
            f"{stat['p95_latency_ms']:.2f}\\,\\text{{ms}} \\\\"
        )
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")

    print(f"[+] Wrote honest recomputed benchmark JSON : {json_path}")
    print(f"[+] Wrote updated LaTeX table              : {tex_path}")
    return ablation_payload


if __name__ == "__main__":
    run_benchmark_matrix()
