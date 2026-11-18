"""
OpenQASM 3.0 Hardware Quantum Assembly Emitter (Driven by QuantumCircuitIR)
===========================================================================
Serializes the exact `QuantumCircuitIR` produced by `compile_paulihedral_schedule`
into OpenQASM 3.0 so simulator and exported QASM execute identical gate sequences.
"""

from __future__ import annotations

from pravaha_q.compiler.circuit_ir import QuantumCircuitIR
from pravaha_q.compiler.paulihedral_router import compile_paulihedral_schedule
from pravaha_q.math_engine.ising_spin_glass import IsingSpinGlass


def circuit_ir_to_openqasm3(
    circuit: QuantumCircuitIR,
    include_dd_pragmas: bool = True,
    header_comment: str = "",
) -> str:
    """Serialize any canonical `QuantumCircuitIR` directly to OpenQASM 3.0."""
    n = circuit.num_qubits
    lines: list[str] = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "",
        "// ===================================================================",
        f"// PravahaQ v0.1.0 — Canonical QuantumCircuitIR Export ({n} Qubits)",
        f"// 1Q Gates: {circuit.count_1q_gates()} | 2Q Gates: {circuit.count_2q_gates()} | Depth: {circuit.depth()}",
    ]
    if header_comment:
        lines.append(f"// {header_comment}")
    lines.append("// ===================================================================")
    lines.append("")

    if include_dd_pragmas:
        lines.append("#pragma ibm_dynamical_decoupling sequence XX")
        lines.append("#pragma pravaha_q compiler vaisheshika_paulihedral_v0_1")
        lines.append("")

    lines.append(f"qubit[{n}] q;")
    lines.append(f"bit[{n}] c;")
    lines.append("")

    for op in circuit.ops:
        g = op.name
        if g == "H":
            lines.append(f"h q[{op.qubits[0]}];")
        elif g == "X":
            lines.append(f"x q[{op.qubits[0]}];")
        elif g in ("RX", "RY", "RZ"):
            lines.append(
                f"{g.lower()}({float(op.params[0]):.6f}) q[{op.qubits[0]}];"
            )
        elif g == "CX":
            lines.append(f"cx q[{op.qubits[0]}], q[{op.qubits[1]}];")
        elif g == "SWAP":
            lines.append(f"swap q[{op.qubits[0]}], q[{op.qubits[1]}];")

    lines.append("")
    for log_idx, phys_idx in enumerate(circuit.measured_qubits):
        lines.append(f"c[{log_idx}] = measure q[{phys_idx}];")
    return "\n".join(lines) + "\n"


def export_openqasm3(
    ising: IsingSpinGlass,
    gamma: float = 0.45,
    beta: float = 0.30,
    include_counterdiabatic: bool = True,
    include_dd_pragmas: bool = True,
) -> str:
    """Compile IsingSpinGlass to an algebraically canceled QuantumCircuitIR and emit OpenQASM 3.0."""
    report = compile_paulihedral_schedule(
        ising,
        include_counterdiabatic=include_counterdiabatic,
        cost_gamma=gamma,
        mixer_beta=beta,
    )
    assert report.optimized_ir is not None
    comment = (
        f"Naive CX: {report.naive_cx_count} -> Algebraic Canceled CX: "
        f"{report.post_cancellation_cx_count} (-{report.cx_reduction_percentage}%)"
    )
    return circuit_ir_to_openqasm3(
        report.optimized_ir,
        include_dd_pragmas=include_dd_pragmas,
        header_comment=comment,
    )
