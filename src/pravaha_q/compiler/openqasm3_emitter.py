"""
OpenQASM 3.0 & 2.0 Hardware Quantum Assembly Emitter
====================================================
Exports a Paulihedral-ordered 3-Local Ising + Counterdiabatic BF-DCQO circuit
into standard OpenQASM 3.0 (with hardware dynamical decoupling pragmas) or
OpenQASM 2.0 for cross-platform execution.
"""

from __future__ import annotations

from pravaha_q.algorithms.agp_commutator import compute_analytical_agp_schedule
from pravaha_q.compiler.paulihedral_router import compile_paulihedral_schedule
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass


def export_openqasm3(
    ising: IsingSpinGlass,
    gamma: float = 0.45,
    beta: float = 0.30,
    include_counterdiabatic: bool = True,
    include_dd_pragmas: bool = True,
) -> str:
    """
    Generate a complete, syntactically valid OpenQASM 3.0 program for the
    Vaisheshika 3-Body HUBO + BF-DCQO impulse circuit.
    """
    n = ising.num_qubits
    h_norm, _, _ = ising.normalized_coefficients()
    pauli_report = compile_paulihedral_schedule(
        ising, include_counterdiabatic=include_counterdiabatic
    )
    agp = compute_analytical_agp_schedule(ising)

    lines: list[str] = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "",
        "// ===================================================================",
        "// PravahaQ v0.1.0 — Vaisheshika 3-Local HUBO & BF-DCQO Circuit",
        f"// Qubits: {n} | 2-Body Couplings: {pauli_report.num_2body_terms} | "
        f"3-Body Cubic: {pauli_report.num_3body_cubic_terms}",
        f"// Paulihedral CX Reduction: {pauli_report.cx_reduction_percentage}%",
        "// ===================================================================",
        "",
    ]
    if include_dd_pragmas:
        lines.append("#pragma ibm_dynamical_decoupling sequence XX")
        lines.append("#pragma pravaha_q compiler vaisheshika_paulihedral_v1")
        lines.append("")

    lines.append(f"qubit[{n}] q;")
    lines.append(f"bit[{n}] c;")
    lines.append("")
    lines.append("// 1. Superposition Initialization")
    for i in range(n):
        lines.append(f"h q[{i}];")

    if include_counterdiabatic:
        lines.append("")
        lines.append(
            f"// 2. Analytical 1st-Order Counterdiabatic AGP Impulse (alpha_1 = {agp.alpha1_coeff})"
        )
        for i, y_ang in sorted(agp.y_1body_angles.items()):
            lines.append(f"ry({2.0 * y_ang:.6f}) q[{i}];")
        for (i, j), yz_ang in sorted(agp.yz_2body_angles.items()):
            lines.append(f"rx(1.570796) q[{i}];")
            lines.append(f"cx q[{i}], q[{j}];")
            lines.append(f"rz({2.0 * yz_ang:.6f}) q[{j}];")
            lines.append(f"cx q[{i}], q[{j}];")
            lines.append(f"rx(-1.570796) q[{i}];")

    lines.append("")
    lines.append("// 3. Paulihedral-Ordered 1-Body, 2-Body & 3-Body Cost Phase Separator")
    for i, h_val in sorted(h_norm.items()):
        lines.append(f"rz({2.0 * gamma * h_val:.6f}) q[{i}];")

    scale = max(ising.spectral_scale, 1e-9)
    for supp, raw_val in pauli_report.ordered_pauli_schedule:
        norm_val = raw_val / scale
        angle = 2.0 * gamma * norm_val
        if len(supp) == 2:
            u, v = supp
            lines.append(f"cx q[{u}], q[{v}];")
            lines.append(f"rz({angle:.6f}) q[{v}];")
            lines.append(f"cx q[{u}], q[{v}];")
        elif len(supp) == 3:
            u, v, w = supp
            lines.append(f"// 3-Body Tryanuka Cubic Interaction Z_{u} Z_{v} Z_{w}")
            lines.append(f"cx q[{u}], q[{v}];")
            lines.append(f"cx q[{v}], q[{w}];")
            lines.append(f"rz({angle:.6f}) q[{w}];")
            lines.append(f"cx q[{v}], q[{w}];")
            lines.append(f"cx q[{u}], q[{v}];")

    lines.append("")
    lines.append("// 4. Transverse Field Mixer Layer")
    for i in range(n):
        lines.append(f"rx({-2.0 * beta:.6f}) q[{i}];")

    lines.append("")
    lines.append("// 5. Computational Basis Readout")
    lines.append("c = measure q;")
    return "\n".join(lines) + "\n"
