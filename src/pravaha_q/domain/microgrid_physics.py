"""3-Phase AC/DC Microgrid power dispatch states, solar/BESS headroom, and thermal inertia."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MicrogridPowerState:
    """
    Discrete operating state for a node's power/cooling envelope.
    For example:
      - State 0 ('Solar_Eco_DVFS'): Low tariff, reduced thermal ceiling.
      - State 1 ('Grid_Turbo_Cool'): Higher tariff, expanded active cooling ceiling.
    """

    state_id: int
    label: str
    energy_cost_tariff: float
    cooling_multiplier: float  # Scales node thermal_capacity_watts (e.g. 0.80 vs 1.25)
    is_eco_throttled: bool = False


@dataclass(frozen=True)
class MicrogridSlice:
    """Instantaneous telemetry slice of the local solar/battery microgrid."""

    power_states: tuple[MicrogridPowerState, ...]
    solar_generation_kw: float = 45.0
    battery_soc_fraction: float = 0.72
    phase_overload_penalty: float = 6.5

    @property
    def num_states(self) -> int:
        return len(self.power_states)
