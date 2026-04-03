# app/core/physics.py
"""
PHYSICS ENGINE — Orbital Mechanics Core
"""

import numpy as np
from typing import Tuple

# ── Constants ──────────────────────────────────────────────────────
MU                   = 398600.4418   # Earth gravitational parameter [km³/s²]
RE                   = 6378.137      # Earth equatorial radius [km]
J2                   = 1.08263e-3    # J2 oblateness coefficient
G0                   = 9.80665e-3    # Standard gravity [km/s²]
ISP                  = 300.0         # Specific impulse [s]
M_DRY                = 500.0         # Dry mass [kg]
M_FUEL_INIT          = 50.0          # Initial fuel [kg]
MAX_DELTA_V          = 0.015         # Max ΔV per burn [km/s] = 15 m/s
THRUSTER_COOLDOWN    = 600           # Cooldown between burns [s]
CONJUNCTION_THRESHOLD = 0.100        # Critical miss distance [km] = 100 m
STATION_KEEPING_RADIUS = 10.0        # Station-keeping box radius [km]
FUEL_EOL_FRACTION    = 0.05          # EOL threshold (5 % of initial)


# ── J2 perturbation ────────────────────────────────────────────────
def j2_acceleration(r: np.ndarray) -> np.ndarray:
    x, y, z  = r
    rn       = np.linalg.norm(r)
    r2       = rn ** 2
    z2       = z ** 2
    factor   = 1.5 * J2 * MU * RE**2 / rn**5
    return np.array([
        factor * x * (5.0 * z2 / r2 - 1.0),
        factor * y * (5.0 * z2 / r2 - 1.0),
        factor * z * (5.0 * z2 / r2 - 3.0),
    ])


def state_derivative(state: np.ndarray) -> np.ndarray:
    r  = state[:3]
    v  = state[3:]
    rn = np.linalg.norm(r)
    a  = -(MU / rn**3) * r + j2_acceleration(r)
    return np.concatenate([v, a])


# ── RK4 integrator ─────────────────────────────────────────────────
def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = state_derivative(state)
    k2 = state_derivative(state + 0.5 * dt * k1)
    k3 = state_derivative(state + 0.5 * dt * k2)
    k4 = state_derivative(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def propagate(state: np.ndarray, duration_s: float, dt: float = 30.0) -> np.ndarray:
    t, cur = 0.0, state.copy()
    while t < duration_s:
        step = min(dt, duration_s - t)
        cur  = rk4_step(cur, step)
        t   += step
    return cur


def propagate_trajectory(state: np.ndarray, duration_s: float, dt: float = 60.0) -> np.ndarray:
    states, t, cur = [state.copy()], 0.0, state.copy()
    while t < duration_s:
        step = min(dt, duration_s - t)
        cur  = rk4_step(cur, step)
        t   += step
        states.append(cur.copy())
    return np.array(states)


# ── RTN frame ──────────────────────────────────────────────────────
def eci_to_rtn_matrix(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    r_hat = r / np.linalg.norm(r)
    h     = np.cross(r, v)
    n_hat = h / np.linalg.norm(h)
    t_hat = np.cross(n_hat, r_hat)
    return np.array([r_hat, t_hat, n_hat])


def rtn_to_eci(dv_rtn: np.ndarray, r: np.ndarray, v: np.ndarray) -> np.ndarray:
    return eci_to_rtn_matrix(r, v).T @ dv_rtn


# ── Fuel / propulsion ──────────────────────────────────────────────
def compute_fuel_consumed(m_current: float, delta_v_km_s: float) -> float:
    return m_current * (1.0 - np.exp(-delta_v_km_s / (ISP * G0)))


def validate_burn(delta_v_vec: np.ndarray, fuel_remaining: float, m_dry: float) -> Tuple[bool, str]:
    dv_mag = np.linalg.norm(delta_v_vec)
    if dv_mag > MAX_DELTA_V:
        return False, f"ΔV {dv_mag*1000:.2f} m/s > 15 m/s limit"
    fuel_needed = compute_fuel_consumed(m_dry + fuel_remaining, dv_mag)
    if fuel_needed > fuel_remaining:
        return False, f"Insufficient fuel: need {fuel_needed:.3f} kg, have {fuel_remaining:.3f} kg"
    return True, "OK"


# ── ECI ↔ geodetic ─────────────────────────────────────────────────
def eci_to_geodetic(r_eci: np.ndarray, gmst_rad: float = 0.0) -> Tuple[float, float, float]:
    x, y, z = r_eci
    rn      = np.linalg.norm(r_eci)
    lat_rad = np.arcsin(z / rn)
    lon_rad = (np.arctan2(y, x) - gmst_rad + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(lat_rad), np.degrees(lon_rad), rn - RE


def compute_gmst(elapsed_seconds: float, epoch_gmst_rad: float = 0.0) -> float:
    return epoch_gmst_rad + 7.292115e-5 * elapsed_seconds


# ── Ground station LOS ─────────────────────────────────────────────
def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r   = RE + alt_km / 1000.0
    return np.array([r*np.cos(lat)*np.cos(lon), r*np.cos(lat)*np.sin(lon), r*np.sin(lat)])


def check_line_of_sight(sat_r_eci, gs_lat, gs_lon, gs_alt_m, gs_min_elev_deg, gmst_rad) -> bool:
    gs_ecef = geodetic_to_ecef(gs_lat, gs_lon, gs_alt_m)
    cg, sg  = np.cos(gmst_rad), np.sin(gmst_rad)
    rot_z   = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
    gs_eci  = rot_z @ gs_ecef
    r_to_sat = sat_r_eci - gs_eci
    dist     = np.linalg.norm(r_to_sat)
    sin_elev = np.dot(r_to_sat, gs_eci / np.linalg.norm(gs_eci)) / dist
    return np.degrees(np.arcsin(np.clip(sin_elev, -1, 1))) >= gs_min_elev_deg


# ── Conjunction (TCA) ──────────────────────────────────────────────
def find_closest_approach(sat_state, deb_state,
                          horizon_s=86400.0, coarse_dt=60.0, fine_dt=5.0) -> Tuple[float, float]:
    sat_s, deb_s = sat_state.copy(), deb_state.copy()
    min_dist, min_t = np.inf, 0.0
    for i in range(1, int(horizon_s / coarse_dt) + 1):
        sat_s = rk4_step(sat_s, coarse_dt)
        deb_s = rk4_step(deb_s, coarse_dt)
        d = np.linalg.norm(sat_s[:3] - deb_s[:3])
        if d < min_dist:
            min_dist, min_t = d, i * coarse_dt

    t_start  = max(0.0, min_t - coarse_dt)
    sat_fine = propagate(sat_state, t_start, dt=coarse_dt)
    deb_fine = propagate(deb_state, t_start, dt=coarse_dt)
    fine_min_dist, fine_min_t = np.inf, t_start
    for j in range(int(2 * coarse_dt / fine_dt) + 1):
        d = np.linalg.norm(sat_fine[:3] - deb_fine[:3])
        if d < fine_min_dist:
            fine_min_dist, fine_min_t = d, t_start + j * fine_dt
        sat_fine = rk4_step(sat_fine, fine_dt)
        deb_fine = rk4_step(deb_fine, fine_dt)
    return fine_min_t, fine_min_dist


# ── Maneuver planning ──────────────────────────────────────────────
def plan_evasion_burn(sat_state, tca_seconds, miss_distance_km) -> np.ndarray:
    if tca_seconds < 1.0:
        tca_seconds = 60.0
    delta_sep = max(0, 0.5 - miss_distance_km)
    dv_t = np.clip(delta_sep / (2.0 * tca_seconds / 1000.0), 0.001, MAX_DELTA_V * 0.8)
    return np.array([0.0, dv_t, 0.0])


def plan_recovery_burn(sat_state, nominal_state) -> np.ndarray:
    dv = nominal_state[3:] - sat_state[3:]
    dv_mag = np.linalg.norm(dv)
    if dv_mag < 1e-6:
        return np.zeros(3)
    if dv_mag > MAX_DELTA_V * 0.9:
        dv = dv * (MAX_DELTA_V * 0.9 / dv_mag)
    return dv


def graveyard_burn(sat_state) -> np.ndarray:
    return np.array([0.0, -MAX_DELTA_V, 0.0])
