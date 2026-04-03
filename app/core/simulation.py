# app/core/simulation.py
"""
SIMULATION STATE MANAGER
"""

import numpy as np
from scipy.spatial import KDTree
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import asyncio

from app.core.physics import (
    propagate, rk4_step,
    find_closest_approach, plan_evasion_burn, plan_recovery_burn,
    rtn_to_eci, compute_fuel_consumed, validate_burn,
    eci_to_geodetic, compute_gmst, check_line_of_sight,
    graveyard_burn,
    CONJUNCTION_THRESHOLD, STATION_KEEPING_RADIUS,
    THRUSTER_COOLDOWN, MAX_DELTA_V, M_DRY, G0, ISP,
)
from app.models.state import (
    SatelliteState, DebrisState, CDMWarning, GroundStation, SatelliteStatus,
)

logger = logging.getLogger(__name__)

GROUND_STATIONS = [
    GroundStation("GS-001", "ISTRAC_Bengaluru",      13.0333,  77.5167,  820, 5.0),
    GroundStation("GS-002", "Svalbard_Sat_Station",  78.2297,  15.4077,  400, 5.0),
    GroundStation("GS-003", "Goldstone_Tracking",    35.4266, -116.890, 1000, 10.0),
    GroundStation("GS-004", "Punta_Arenas",         -53.1500,  -70.917,   30, 5.0),
    GroundStation("GS-005", "IIT_Delhi_Ground_Node", 28.5450,   77.193,  225, 15.0),
    GroundStation("GS-006", "McMurdo_Station",      -77.8463,  166.668,   10, 5.0),
]


class SimulationManager:
    def __init__(self):
        self.sim_time:  float    = 0.0
        # FIX: keep epoch timezone-naive so arithmetic with timedelta works everywhere
        self.epoch: datetime     = datetime(2026, 3, 12, 8, 0, 0)
        self.satellites: Dict[str, SatelliteState] = {}
        self.debris:     Dict[str, DebrisState]    = {}
        self.active_cdms: List[CDMWarning]         = []
        self._debris_tree: Optional[KDTree]        = None
        self._debris_ids:  List[str]               = []
        self._tree_dirty:  bool                    = True
        self._lock = asyncio.Lock()
        self.total_collisions       = 0
        self.total_maneuvers_executed = 0
        self._initialize_default_constellation()

    # ── Walker-Delta 50-sat constellation ─────────────────────────
    def _initialize_default_constellation(self):
        n_planes, n_per_plane = 5, 10
        alt_km, inc_deg = 550.0, 53.0
        a      = 6378.137 + alt_km
        v_circ = np.sqrt(398600.4418 / a)
        for plane in range(n_planes):
            raan = np.radians(plane * 72.0)
            inc  = np.radians(inc_deg)
            for slot in range(n_per_plane):
                ta    = np.radians(slot * 36.0)
                r_orb = np.array([a*np.cos(ta), a*np.sin(ta), 0.0])
                v_orb = np.array([-v_circ*np.sin(ta), v_circ*np.cos(ta), 0.0])
                Rx = np.array([[1,0,0],[0,np.cos(inc),-np.sin(inc)],[0,np.sin(inc),np.cos(inc)]])
                Rz = np.array([[np.cos(raan),-np.sin(raan),0],[np.sin(raan),np.cos(raan),0],[0,0,1]])
                state = np.concatenate([Rz @ Rx @ r_orb, Rz @ Rx @ v_orb])
                sid   = f"SAT-P{plane+1}-{slot+1:02d}"
                self.satellites[sid] = SatelliteState(sid, state)
        logger.info(f"Initialized {len(self.satellites)} satellites")

    # ── KD-Tree ───────────────────────────────────────────────────
    def _rebuild_debris_tree(self):
        if not self.debris:
            self._debris_tree, self._debris_ids = None, []
            return
        self._debris_ids = list(self.debris.keys())
        positions = np.array([self.debris[d].state[:3] for d in self._debris_ids])
        self._debris_tree = KDTree(positions)
        self._tree_dirty  = False

    def _get_nearby_debris(self, sat_pos: np.ndarray, radius_km: float = 500.0) -> List[str]:
        if self._tree_dirty:
            self._rebuild_debris_tree()
        if self._debris_tree is None:
            return []
        return [self._debris_ids[i] for i in self._debris_tree.query_ball_point(sat_pos, radius_km)]

    # ── Telemetry ingestion ────────────────────────────────────────
    async def ingest_telemetry(self, timestamp: datetime, objects: list) -> int:
        processed, debris_updated = 0, False
        async with self._lock:
            for obj in objects:
                sv = np.array([obj.r.x, obj.r.y, obj.r.z, obj.v.x, obj.v.y, obj.v.z])
                if obj.type == "SATELLITE":
                    if obj.id in self.satellites:
                        self.satellites[obj.id].state = sv
                    else:
                        self.satellites[obj.id] = SatelliteState(obj.id, sv)
                else:
                    if obj.id in self.debris:
                        self.debris[obj.id].state = sv
                    else:
                        self.debris[obj.id] = DebrisState(obj.id, sv)
                    debris_updated = True
                processed += 1
            if debris_updated:
                self._tree_dirty = True
        return processed

    # ── Conjunction assessment ─────────────────────────────────────
    async def run_conjunction_assessment(self, horizon_s: float = 86400.0):
        if self._tree_dirty:
            self._rebuild_debris_tree()
        new_cdms = []
        existing = {(c.sat_id, c.deb_id) for c in self.active_cdms}
        for sid, sat in self.satellites.items():
            if sat.status == SatelliteStatus.DEAD:
                continue
            for did in self._get_nearby_debris(sat.state[:3]):
                if (sid, did) in existing:
                    continue
                tca_s, miss = find_closest_approach(sat.state, self.debris[did].state, horizon_s)
                if miss < CONJUNCTION_THRESHOLD:
                    cdm = CDMWarning(sid, did, self.sim_time + tca_s, miss)
                    new_cdms.append(cdm)
                    logger.warning(f"CDM: {sid} ↔ {did} miss={miss*1000:.1f}m TCA+{tca_s/60:.1f}min")
        async with self._lock:
            self.active_cdms = [c for c in self.active_cdms if c.tca_sim_time > self.sim_time]
            self.active_cdms.extend(new_cdms)
        for cdm in new_cdms:
            if cdm.is_critical and not cdm.evasion_scheduled:
                await self._auto_schedule_evasion(cdm)
        return len(self.active_cdms)

    async def _auto_schedule_evasion(self, cdm: CDMWarning):
        sat = self.satellites.get(cdm.sat_id)
        if not sat or sat.is_eol:
            return
        gmst = compute_gmst(self.sim_time)
        has_los = any(
            check_line_of_sight(sat.state[:3], gs.lat, gs.lon, gs.alt_m, gs.min_elev_deg, gmst)
            for gs in GROUND_STATIONS
        )
        if not has_los:
            return
        cooldown_rem = 0.0
        if sat.last_burn_time is not None:
            cooldown_rem = max(0.0, THRUSTER_COOLDOWN - (self.sim_time - sat.last_burn_time))
        burn1_time = self.sim_time + 10.0 + cooldown_rem
        tca_s = cdm.tca_sim_time - self.sim_time
        dv_rtn = plan_evasion_burn(sat.state, tca_s, cdm.miss_distance_km)
        dv_eci = rtn_to_eci(dv_rtn, sat.state[:3], sat.state[3:])
        valid, reason = validate_burn(dv_eci, sat.m_fuel, sat.m_dry)
        if not valid:
            logger.error(f"Evasion burn invalid for {cdm.sat_id}: {reason}")
            return
        burn2_time = cdm.tca_sim_time + THRUSTER_COOLDOWN + 60.0
        dv_rec = plan_recovery_burn(
            propagate(sat.state, burn2_time - self.sim_time),
            propagate(sat.nominal_state, burn2_time - self.sim_time),
        )
        async with self._lock:
            sat.scheduled_burns.append({"burn_id": f"EVA_{cdm.deb_id}", "burn_time": burn1_time, "dv_eci": dv_eci,  "type": "EVASION"})
            sat.scheduled_burns.append({"burn_id": f"REC_{cdm.deb_id}", "burn_time": burn2_time, "dv_eci": dv_rec,  "type": "RECOVERY"})
            sat.status = SatelliteStatus.EVADING
            cdm.evasion_scheduled = True

    # ── Maneuver scheduling ────────────────────────────────────────
    async def schedule_maneuver(self, sat_id: str, burn_sequence: list) -> Tuple[bool, float]:
        sat = self.satellites.get(sat_id)
        if not sat:
            return False, 0.0
        gmst = compute_gmst(self.sim_time)
        has_los = any(
            check_line_of_sight(sat.state[:3], gs.lat, gs.lon, gs.alt_m, gs.min_elev_deg, gmst)
            for gs in GROUND_STATIONS
        )
        m_remaining   = sat.m_total
        last_burn_time = sat.last_burn_time or (self.sim_time - THRUSTER_COOLDOWN)
        for burn in burn_sequence:
            # FIX: strip timezone info from burnTime before subtracting naive epoch
            bt = burn.burnTime.replace(tzinfo=None)
            burn_epoch = (bt - self.epoch).total_seconds()
            if burn_epoch - last_burn_time < THRUSTER_COOLDOWN:
                logger.warning(f"Burn {burn.burn_id} violates cooldown")
                return False, m_remaining
            dv = burn.deltaV_vector.to_numpy()
            valid, reason = validate_burn(dv, m_remaining - M_DRY, M_DRY)
            if not valid:
                return False, m_remaining
            m_remaining   -= compute_fuel_consumed(m_remaining, np.linalg.norm(dv))
            last_burn_time = burn_epoch
        async with self._lock:
            for burn in burn_sequence:
                bt = burn.burnTime.replace(tzinfo=None)
                burn_epoch = (bt - self.epoch).total_seconds()
                sat.scheduled_burns.append({
                    "burn_id":   burn.burn_id,
                    "burn_time": burn_epoch,
                    "dv_eci":    burn.deltaV_vector.to_numpy(),
                    "type":      "MANUAL",
                })
            sat.scheduled_burns.sort(key=lambda b: b["burn_time"])
        return has_los, m_remaining

    # ── Simulation step ────────────────────────────────────────────
    async def step(self, step_seconds: float) -> Tuple[int, int]:
        collisions, maneuvers = 0, 0
        end_time, SUB_STEP, t = self.sim_time + step_seconds, 30.0, self.sim_time
        while t < end_time:
            dt = min(SUB_STEP, end_time - t)
            async with self._lock:
                for deb in self.debris.values():
                    deb.state = rk4_step(deb.state, dt)
                for sid, sat in self.satellites.items():
                    if sat.status == SatelliteStatus.DEAD:
                        continue
                    burns = sorted([b for b in sat.scheduled_burns if t <= b["burn_time"] < t+dt],
                                   key=lambda b: b["burn_time"])
                    for burn in burns:
                        dt_to = burn["burn_time"] - t
                        if dt_to > 0:
                            sat.state = propagate(sat.state, dt_to)
                        dv = burn["dv_eci"]
                        sat.state[3:]   += dv
                        sat.m_fuel      -= compute_fuel_consumed(sat.m_total, np.linalg.norm(dv))
                        sat.last_burn_time = burn["burn_time"]
                        sat.scheduled_burns.remove(burn)
                        maneuvers += 1
                        if burn["type"] == "RECOVERY":
                            sat.status = SatelliteStatus.NOMINAL
                    sat.state         = rk4_step(sat.state, dt)
                    sat.nominal_state = rk4_step(sat.nominal_state, dt)
                    if np.linalg.norm(sat.state[:3] - sat.nominal_state[:3]) > STATION_KEEPING_RADIUS:
                        sat.status = SatelliteStatus.RECOVERING
                        sat.outage_seconds += dt
                    for deb in self.debris.values():
                        if np.linalg.norm(sat.state[:3] - deb.state[:3]) < CONJUNCTION_THRESHOLD:
                            collisions += 1
                            sat.collision_count += 1
                    if sat.is_eol and sat.status != SatelliteStatus.EOL:
                        sat.status = SatelliteStatus.EOL
                        dv_rtn = graveyard_burn(sat.state)
                        dv_eci = rtn_to_eci(dv_rtn, sat.state[:3], sat.state[3:])
                        sat.scheduled_burns.append({
                            "burn_id": f"GRAVE_{sid}", "burn_time": t+dt+10.0,
                            "dv_eci": dv_eci, "type": "EOL",
                        })
            t += dt
        self.sim_time = end_time
        self.total_collisions          += collisions
        self.total_maneuvers_executed  += maneuvers
        self._tree_dirty = True
        await self.run_conjunction_assessment()
        return collisions, maneuvers

    # ── Snapshot ───────────────────────────────────────────────────
    def get_snapshot(self) -> dict:
        gmst = compute_gmst(self.sim_time)
        sats_out = []
        for sid, sat in self.satellites.items():
            lat, lon, alt = eci_to_geodetic(sat.state[:3], gmst)
            sats_out.append({
                "id":      sid,
                "lat":     round(lat, 4),
                "lon":     round(lon, 4),
                "fuel_kg": round(sat.m_fuel, 3),
                "status":  sat.status.value,
            })
        deb_out = []
        for did, deb in self.debris.items():
            lat, lon, alt = eci_to_geodetic(deb.state[:3], gmst)
            deb_out.append([did, round(lat, 3), round(lon, 3), round(alt, 1)])
        return {
            "timestamp":   (self.epoch + timedelta(seconds=self.sim_time)).isoformat() + "Z",
            "satellites":  sats_out,
            "debris_cloud": deb_out,
        }


# ── Singleton ──────────────────────────────────────────────────────
_sim: Optional[SimulationManager] = None

def get_sim_manager() -> SimulationManager:
    global _sim
    if _sim is None:
        _sim = SimulationManager()
    return _sim
