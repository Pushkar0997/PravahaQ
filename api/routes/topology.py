"""Endpoints for generating and inspecting Edge-AI Cluster & Microgrid topologies."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter

from api.schemas import ProblemSpecRequest
from pravaha_q.domain.problem_generator import generate_coupled_problem

router = APIRouter(prefix="/topology", tags=["1. Digital Twin & Topology"])


@router.post("/generate")
def generate_digital_twin(req: ProblemSpecRequest) -> dict[str, Any]:
    """Synthesize a deterministic Coupled Edge-AI + 3-Phase Microgrid digital twin."""
    prob = generate_coupled_problem(
        num_nodes=req.num_nodes,
        num_tasks=req.num_tasks,
        num_power_states=req.num_power_states,
        seed=req.seed,
    )
    return {
        "problem_name": prob.name,
        "total_binary_variables": prob.total_binary_variables,
        "num_routing_vars": prob.num_routing_vars,
        "num_power_vars": prob.num_power_vars,
        "variable_labels": prob.variable_labels(),
        "nodes": [asdict(n) for n in prob.cluster.nodes],
        "interconnect_latency_ms": prob.cluster.interconnect_latency_ms,
        "tasks": [asdict(t) for t in prob.workload.tasks],
        "kv_sync_weights": prob.workload.kv_sync_weights,
        "power_states": [asdict(p) for p in prob.microgrid.power_states],
    }
