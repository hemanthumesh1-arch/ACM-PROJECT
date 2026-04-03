# app/api/routes.py
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timedelta
import logging

from app.models.state import SatelliteStatus
from app.core.simulation import SimulationManager, get_sim_manager
from pydantic import BaseModel
from typing import List
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Pydantic request/response models ──────────────────────────────

class Vec3(BaseModel):
    x: float
    y: float
    z: float

    def to_numpy(self):
        return np.array([self.x, self.y, self.z])


class TelemetryObject(BaseModel):
    id:   str
    type: str
    r:    Vec3
    v:    Vec3


class TelemetryRequest(BaseModel):
    timestamp: datetime
    objects:   List[TelemetryObject]


class BurnCommand(BaseModel):
    burn_id:        str
    burnTime:       datetime
    deltaV_vector:  Vec3


class ManeuverRequest(BaseModel):
    satelliteId:        str
    maneuver_sequence:  List[BurnCommand]


class SimStepRequest(BaseModel):
    step_seconds: float


# ── POST /api/telemetry ────────────────────────────────────────────

@router.post("/api/telemetry")
async def ingest_telemetry(req: TelemetryRequest,
                           sim: SimulationManager = Depends(get_sim_manager)):
    processed = await sim.ingest_telemetry(req.timestamp, req.objects)
    return {
        "status":            "ACK",
        "processed_count":   processed,
        "active_cdm_warnings": len(sim.active_cdms),
    }


# ── POST /api/maneuver/schedule ────────────────────────────────────

@router.post("/api/maneuver/schedule", status_code=202)
async def schedule_maneuver(req: ManeuverRequest,
                            sim: SimulationManager = Depends(get_sim_manager)):
    sat = sim.satellites.get(req.satelliteId)
    if not sat:
        raise HTTPException(404, f"Satellite {req.satelliteId} not found")
    if sat.status == SatelliteStatus.DEAD:
        raise HTTPException(400, f"Satellite {req.satelliteId} is DEAD")

    has_los, projected_mass = await sim.schedule_maneuver(req.satelliteId, req.maneuver_sequence)
    if projected_mass <= 0:
        raise HTTPException(400, "Maneuver validation failed")

    return {
        "status": "SCHEDULED",
        "validation": {
            "ground_station_los":          has_los,
            "sufficient_fuel":             True,
            "projected_mass_remaining_kg": round(projected_mass, 3),
        },
    }


# ── POST /api/simulate/step ───────────────────────────────────────

@router.post("/api/simulate/step")
async def simulate_step(req: SimStepRequest,
                        sim: SimulationManager = Depends(get_sim_manager)):
    collisions, maneuvers = await sim.step(req.step_seconds)
    new_ts = sim.epoch + timedelta(seconds=sim.sim_time)
    return {
        "status":              "STEP_COMPLETE",
        "new_timestamp":       new_ts.isoformat() + "Z",
        "collisions_detected": collisions,
        "maneuvers_executed":  maneuvers,
    }


# ── GET /api/visualization/snapshot ──────────────────────────────

@router.get("/api/visualization/snapshot")
async def get_snapshot(sim: SimulationManager = Depends(get_sim_manager)):
    return sim.get_snapshot()


# ── GET /health ────────────────────────────────────────────────────

@router.get("/health")
async def health_check(sim: SimulationManager = Depends(get_sim_manager)):
    return {
        "status":          "healthy",
        "sim_time_s":      sim.sim_time,
        "satellites":      len(sim.satellites),
        "debris":          len(sim.debris),
        "active_cdms":     len(sim.active_cdms),
        "total_collisions": sim.total_collisions,
        "total_maneuvers": sim.total_maneuvers_executed,
    }


# ── GET /api/debug/satellites ──────────────────────────────────────

@router.get("/api/debug/satellites")
async def list_satellites(sim: SimulationManager = Depends(get_sim_manager)):
    return {
        "satellites": [
            {
                "id":              sid,
                "status":          sat.status.value,
                "fuel_kg":         round(sat.m_fuel, 3),
                "fuel_pct":        round(sat.fuel_fraction * 100, 1),
                "position_km":     sat.state[:3].tolist(),
                "velocity_km_s":   sat.state[3:].tolist(),
                "scheduled_burns": len(sat.scheduled_burns),
                "collisions":      sat.collision_count,
                "outage_seconds":  round(sat.outage_seconds, 1),
            }
            for sid, sat in sim.satellites.items()
        ],
        "count": len(sim.satellites),
    }


# ── GET /api/debug/cdms ────────────────────────────────────────────

@router.get("/api/debug/cdms")
async def list_cdms(sim: SimulationManager = Depends(get_sim_manager)):
    return {
        "active_cdms": [
            {
                "sat_id":            c.sat_id,
                "deb_id":            c.deb_id,
                "tca_sim_time":      c.tca_sim_time,
                "miss_distance_m":   round(c.miss_distance_km * 1000, 1),
                "is_critical":       c.is_critical,
                "evasion_scheduled": c.evasion_scheduled,
            }
            for c in sim.active_cdms
        ],
        "count": len(sim.active_cdms),
    }
