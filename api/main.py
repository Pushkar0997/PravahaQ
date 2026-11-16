"""
PravahaQ v0.1.0 — FastAPI Production Control Plane Entrypoint
=============================================================
Exposes REST APIs for Digital Twin synthesis, Vaisheshika 3-Body HUBO/Ising
compilation, BF-DCQO / Multistage PCE / IQO-BnB execution, OpenQASM 3.0 &
NVIDIA CUDA-Q code generation, and Prometheus observability.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from api.routes import benchmarks, compiler, optimize, topology
from pravaha_q.infrastructure.observability import GLOBAL_TELEMETRY
from pravaha_q.version import __compiler_codename__, __version__

app = FastAPI(
    title="PravahaQ Control Plane API",
    description=(
        "Hybrid HPC-Quantum Optimization Engine for Coupled Edge-AI Inference "
        "Routing and 3-Phase Microgrid Thermal Dispatch."
    ),
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(topology.router, prefix="/api/v1")
app.include_router(compiler.router, prefix="/api/v1")
app.include_router(optimize.router, prefix="/api/v1")
app.include_router(benchmarks.router, prefix="/api/v1")


@app.get("/api/v1/health", tags=["0. System & Observability"])
def health_probe() -> dict[str, str | int]:
    """Kubernetes liveness and readiness probe."""
    return {
        "status": "healthy",
        "service": "pravaha-q-control-plane",
        "version": __version__,
        "compiler": __compiler_codename__,
        "jobs_executed": GLOBAL_TELEMETRY.total_jobs_executed,
    }


@app.get(
    "/api/v1/metrics",
    response_class=PlainTextResponse,
    tags=["0. System & Observability"],
)
def prometheus_metrics() -> str:
    """Prometheus scrape endpoint."""
    return GLOBAL_TELEMETRY.to_prometheus_text()
