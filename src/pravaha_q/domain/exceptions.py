"""Domain-specific exception hierarchy for PravahaQ."""

from __future__ import annotations


class PravahaError(Exception):
    """Base exception for all PravahaQ engine errors."""


class TopologyValidationError(PravahaError):
    """Raised when an Edge-AI cluster or microgrid topology is physically invalid."""


class HUBOCompilationError(PravahaError):
    """Raised when the Vaisheshika HUBO compiler encounters inconsistent dimensions."""


class QubitCompressionCapacityError(PravahaError):
    """Raised when Pauli Correlation Encoding (PCE) lacks enough correlators for N variables."""


class FeasibilityRepairError(PravahaError):
    """Raised if a problem instance has zero physically valid schedules."""
