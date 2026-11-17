# Contributing to PravahaQ

Thank you for your interest in contributing to **PravahaQ** (`pravaha_q` / compiler codename `Vaisheshika-3Local`), our hybrid HPC–Quantum optimization engine and control plane for coupled Edge-AI inference routing and 3-phase microgrid thermal dispatch.

---

## 🧭 Engineering & Spec-Driven Development Standards

PravahaQ follows strict **Spec-Driven Development (SDD)** and mathematical verification gates:

1. **Mathematical & Physical Fidelity First**:
   - Every Hamiltonian term in `src/pravaha_q/core/hubo_builder.py` (1-body $\mathcal{H}_1$, 2-body $\mathcal{H}_2$, and 3-body $\mathcal{H}_3$ Vaisheshika terms) must match exact finite-difference or brute-force bitstring evaluations to within $10^{-10}$ tolerance on small instances ($N \le 14$).
   - **Strict Validator Separation**: Physical schedule feasibility (`ScheduleValidator` returning `VALID` / `INVALID` over one-hot assignment, node memory/compute capacity, latency SLA, and 3-phase thermal limits) must remain strictly decoupled from optimizer execution status (`SolverStatus`).
   - **No Silent Stubs**: Do not submit mock random bitstring generators disguised as quantum or tensor-network solvers. If a backend SDK (`qiskit`, `pennylane`, `cirq`) is an optional dependency, raise an explicit `MissingOptionalDependencyError` or fall back transparently with an explicit warning flag in telemetry.

2. **Branching & Pull Request Workflow**:
   - Create focused feature or fix branches from `main` (e.g., `fix/hubo-r1-constraint`, `feat/schedule-validator`, `docs/adr-update`).
   - Keep pull requests scoped to a single logical component so mathematical and unit tests can be audited cleanly.

---

## 🧪 Local Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/Pushkar0997/PravahaQ.git
cd PravahaQ

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install core package + dev/test dependencies
pip install -e ".[dev]"
```

### Running the Verification Suite

Before opening a Pull Request, verify that all unit, property-based (`hypothesis`), and linting checks pass:

```bash
# Run pytest suite
pytest -v

# Run ruff linter & formatter check
ruff check src/ api/ tests/
```

---

## 📜 License

By contributing to PravahaQ, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
