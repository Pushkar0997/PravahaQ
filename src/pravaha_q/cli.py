"""PravahaQ Rich Command-Line Interface."""

from __future__ import annotations

import argparse
import sys

from pravaha_q.domain.problem_generator import generate_coupled_problem
from pravaha_q.math_engine.hubo_builder import VaisheshikaHUBOCompiler
from pravaha_q.math_engine.ising_spin_glass import hubo_to_ising
from pravaha_q.solvers.qiskit_adapter import QiskitBFDCQOSolver


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PravahaQ v0.1.0 Quantum Optimization CLI"
    )
    subparsers = parser.add_subparsers(dest="command")

    opt_parser = subparsers.add_parser(
        "optimize", help="Run coupled Edge-AI + Microgrid optimization"
    )
    opt_parser.add_argument("--nodes", type=int, default=3)
    opt_parser.add_argument("--tasks", type=int, default=4)
    opt_parser.add_argument("--algorithm", type=str, default="bf_dcqo")

    args = parser.parse_args()
    if args.command == "optimize":
        print(
            f"[*] Running PravahaQ optimization for {args.nodes} nodes, {args.tasks} tasks..."
        )
        prob = generate_coupled_problem(num_nodes=args.nodes, num_tasks=args.tasks)
        hubo = VaisheshikaHUBOCompiler().compile(prob)
        ising = hubo_to_ising(hubo)
        res = QiskitBFDCQOSolver().solve(ising)
        print(
            f"[+] Success! Best Bitstring: {res.best_bitstring} | "
            f"Energy: {res.best_energy} | Time: {res.execution_time_ms}ms"
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
