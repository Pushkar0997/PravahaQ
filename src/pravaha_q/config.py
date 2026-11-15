"""Centralized configuration and hardware calibration loader for PravahaQ."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class UnbalancedPenaltyConfig:
    """Hyperparameters for Montanez-Barrera slack-free inequality penalization."""

    lambda1_linear: float = 3.5
    lambda2_quadratic: float = 6.0
    equality_weight: float = 10.0


@dataclass(frozen=True)
class BFDCQOConfig:
    """Hyperparameters for Bias-Field Digitized Counterdiabatic Optimization."""

    max_iterations: int = 4
    initial_bias_gain: float = 0.65
    bias_sharpness_gamma: float = 2.5
    agp_commutator_order: int = 1
    impulse_dt: float = 0.45
    polarization_stop_threshold: float = 0.92


@dataclass(frozen=True)
class MitigationConfig:
    """Configuration for ZNE, TREX readout mitigation, and CVaR tail aggregation."""

    enable_trex_readout: bool = True
    enable_zne_extrapolation: bool = True
    zne_scale_factors: tuple[float, ...] = (1.0, 3.0, 5.0)
    cvar_alpha: float = 0.20
    default_shots: int = 2048


@dataclass(frozen=True)
class PravahaEngineConfig:
    """Master immutable configuration for the PravahaQ engine."""

    env: str = field(default_factory=lambda: os.getenv("PRAVAHA_ENV", "development"))
    default_algorithm: str = field(
        default_factory=lambda: os.getenv("PRAVAHA_DEFAULT_ALGORITHM", "bf_dcqo")
    )
    default_sdk: str = field(
        default_factory=lambda: os.getenv("PRAVAHA_DEFAULT_SDK", "qiskit")
    )
    random_seed: int = field(
        default_factory=lambda: int(os.getenv("PRAVAHA_RANDOM_SEED", "42"))
    )
    sla_timeout_ms: float = 450.0
    enable_cubic_3body_terms: bool = True
    guarantee_feasibility_repair: bool = True
    penalties: UnbalancedPenaltyConfig = field(default_factory=UnbalancedPenaltyConfig)
    bf_dcqo: BFDCQOConfig = field(default_factory=BFDCQOConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> PravahaEngineConfig:
        """Load engine configuration from a YAML file with safe fallback defaults."""
        path = Path(yaml_path)
        if not path.exists():
            return cls()

        with path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}

        engine_sec = raw.get("engine", {})
        prob_sec = raw.get("default_problem", {})
        algo_sec = raw.get("algorithms", {})
        bf_sec = algo_sec.get("bf_dcqo", {})
        mit_sec = raw.get("mitigation", {})

        penalties = UnbalancedPenaltyConfig(
            lambda1_linear=float(prob_sec.get("unbalanced_penalty_lambda1", 3.5)),
            lambda2_quadratic=float(prob_sec.get("unbalanced_penalty_lambda2", 6.0)),
            equality_weight=float(prob_sec.get("equality_penalty_weight", 10.0)),
        )
        bf_dcqo = BFDCQOConfig(
            max_iterations=int(bf_sec.get("max_iterations", 4)),
            initial_bias_gain=float(bf_sec.get("initial_bias_gain", 0.65)),
            bias_sharpness_gamma=float(bf_sec.get("bias_sharpness_gamma", 2.5)),
            agp_commutator_order=int(bf_sec.get("agp_commutator_order", 1)),
            impulse_dt=float(bf_sec.get("impulse_dt", 0.45)),
            polarization_stop_threshold=float(bf_sec.get("polarization_stop_threshold", 0.92)),
        )
        mitigation = MitigationConfig(
            enable_trex_readout=bool(mit_sec.get("enable_trex_readout", True)),
            enable_zne_extrapolation=bool(mit_sec.get("enable_zne_extrapolation", True)),
            zne_scale_factors=tuple(
                float(x) for x in mit_sec.get("zne_scale_factors", [1.0, 3.0, 5.0])
            ),
            cvar_alpha=float(mit_sec.get("cvar_alpha", 0.20)),
            default_shots=int(mit_sec.get("default_shots", 2048)),
        )

        return cls(
            default_algorithm=str(algo_sec.get("default", "bf_dcqo")),
            random_seed=int(engine_sec.get("random_seed", 42)),
            sla_timeout_ms=float(engine_sec.get("sla_timeout_ms", 450.0)),
            enable_cubic_3body_terms=bool(prob_sec.get("enable_cubic_3body_terms", True)),
            guarantee_feasibility_repair=bool(
                engine_sec.get("guarantee_feasibility_repair", True)
            ),
            penalties=penalties,
            bf_dcqo=bf_dcqo,
            mitigation=mitigation,
        )
