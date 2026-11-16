"""End-to-end integration tests for the FastAPI Control Plane."""

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


def test_compiler_and_optimize_endpoints() -> None:
    comp_resp = client.post(
        "/api/v1/compiler/inspect",
        json={"num_nodes": 2, "num_tasks": 2, "num_power_states": 1},
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
            "num_power_states": 1,
            "algorithm": "bf_dcqo",
            "sdk": "qiskit",
        },
    )
    assert opt_resp.status_code == 200
    opt_data = opt_resp.json()
    assert opt_data["schedule"]["is_feasible"] is True
    assert opt_data["approximation_ratio"] >= 0.80
