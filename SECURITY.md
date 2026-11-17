# Security Policy

## Supported Versions

PravahaQ is currently under active development in the `0.1.x` clean-core rebuild series. Security updates are applied to the latest commit on `main`.

| Version | Supported |
| :--- | :--- |
| `0.1.x` (`main`) | ✅ Supported |
| `< 0.1.0` | ❌ Not Supported |

---

## 🔒 Security & Operational Scope

PravahaQ includes both a mathematical optimization core (`src/pravaha_q/`) and a FastAPI control plane (`api/`) with optional cloud QPU provider integrations (IBM Quantum Runtime).

1. **API Keys & Cloud QPU Credentials**:
   - Never commit `IBM_QUANTUM_TOKEN`, API keys, or `.env` files to version control.
   - Always use `.env.example` as a template and pass secrets via environment variables at runtime.
2. **Control Plane Input Validation**:
   - All problem dimensions ($T, M, N, R$) submitted to the FastAPI endpoints (`/api/v1/optimize`, `/api/v1/compile`) are bounded by strict Pydantic schemas to prevent exponential $O(2^{N_{\text{qubits}}})$ memory exhaustion on classical statevector or exact diagonalization code paths.

---

## 📬 Reporting a Vulnerability

If you discover a security vulnerability, unbounded resource exhaustion vector, or credential handling issue in PravahaQ:

- **Do NOT open a public GitHub issue.**
- Please report it privately via email to **Pushkar Kumar** at **[pushkarkumar0997@gmail.com](mailto:pushkarkumar0997@gmail.com)** with the subject line `[SECURITY][PravahaQ] Brief description`.
- Include steps to reproduce, affected files/endpoints, and potential impact. You will receive an initial response within **48 hours**.
