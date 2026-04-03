# app/models/state.py
from enum import Enum
import numpy as np


class SatelliteStatus(str, Enum):
    NOMINAL    = "NOMINAL"
    EVADING    = "EVADING"
    RECOVERING = "RECOVERING"
    EOL        = "EOL"
    DEAD       = "DEAD"


class SatelliteState:
    def __init__(self, sat_id: str, state: np.ndarray):
        self.id             = sat_id
        self.state          = state.copy()
        self.nominal_state  = state.copy()   # FIX: was missing — caused AttributeError in simulation step
        self.m_fuel         = 50.0
        self.m_dry          = 500.0
        self.status         = SatelliteStatus.NOMINAL
        self.scheduled_burns = []
        self.last_burn_time = None
        self.collision_count = 0
        self.outage_seconds  = 0.0

    @property
    def m_total(self) -> float:
        """FIX: was missing — simulation.py calls sat.m_total"""
        return self.m_dry + self.m_fuel

    @property
    def fuel_fraction(self) -> float:
        return self.m_fuel / 50.0

    @property
    def is_eol(self) -> bool:
        """FIX: was missing — simulation.py checks sat.is_eol"""
        return self.fuel_fraction <= 0.05


class DebrisState:
    def __init__(self, deb_id: str, state: np.ndarray):
        self.id    = deb_id
        self.state = state.copy()


class CDMWarning:
    def __init__(self, sat_id, deb_id, tca_time, miss_distance_km):
        self.sat_id           = sat_id
        self.deb_id           = deb_id
        self.tca_sim_time     = tca_time
        self.miss_distance_km = miss_distance_km
        self.is_critical      = miss_distance_km < 0.1
        self.evasion_scheduled = False


class GroundStation:
    def __init__(self, gs_id, name, lat, lon, alt_m, min_elev_deg):
        self.id           = gs_id
        self.name         = name
        self.lat          = lat
        self.lon          = lon
        self.alt_m        = alt_m
        self.min_elev_deg = min_elev_deg
