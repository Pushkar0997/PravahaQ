"""End-to-end integration tests for the FastAPI Control Plane & Admission Guards."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_and_prometheus_metrics() -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

    m_resp = client.get("/api/v1/metrics")
    assert m_resp.status_code == 200
    assert "pravaha_q_jobs_total" in m_resp.text


def test_compiler_and_optimize_endpoints_with_honest_validation() -> None:
    comp_resp = client.post(
        "/api/v1/compiler/inspect",
        json={"num_nodes": 2, "num_tasks": 2, "num_power_states": 2},
    )
    assert comp_resp.status_code == 200
    comp_data = comp_resp.json()
    assert "OPENQASM 3.0;" in comp_data["openqasm3_source"]
    assert "@cudaq.kernel" in comp_data["cudaq_kernel_source"]

    opt_resp = client.post(
        "/api/v1/optimize/run",
        json={
            "num_nodes": 2,
            "num_tasks": 3,
            "num_power_states": 2,
            "algorithm": "bf_dcqo",
            "sdk": "qiskit",
        },
    )
    assert opt_resp.status_code == 200
    opt_data = opt_resp.json()
    assert opt_data["schedule"]["is_feasible"] is True
    assert opt_data["schedule"]["validity"] == "VALID"
    assert opt_data["operational_cost_f"] is not None
    assert opt_data["execution_mode"] == "SHOT_SIMULATION"


def test_api_rejects_incompatible_backend_and_flags_overloaded_instances() -> None:
    """Verify Step 8 compatibility rejection (HTTP 422) and Step 7 overloaded rejection."""
    # 1. Incompatible algorithm/backend pair: `exact` on `pennylane`
    bad_resp = client.post(
        "/api/v1/optimize/run",
        json={
            "num_nodes": 2,
            "num_tasks": 2,
            "num_power_states": 1,
            "algorithm": "exact",
            "sdk": "pennylane",
        },
    )
    assert bad_resp.status_code == 422

    # 2. Deliberately overloaded instance (`capacity_scale=0.15`) -> PROVEN_INFEASIBLE
    inf_resp = client.post(
        "/api/v1/optimize/run",
        json={
            "num_nodes": 2,
            "num_tasks": 3,
            "num_power_states": 1,
            "capacity_scale": 0.15,
            "algorithm": "bf_dcqo",
            "sdk": "qiskit",
        },
    )
    assert inf_resp.status_code == 200
    inf_data = inf_resp.json()
    assert inf_data["schedule"]["is_feasible"] is False
    assert inf_data["schedule"]["validity"] == "INVALID"
    assert inf_data["schedule"]["solver_status"] == "PROVEN_INFEASIBLE"
