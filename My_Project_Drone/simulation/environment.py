"""
Phase 3.4 — Simulation Environment

Provides:
  - AirSim world obstacle positions as GPS coords (mapped from Blocks NED metres)
  - VictimTrajectory: deterministic + noisy GPS path generator
  - WeatherPreset: enum of AirSim weather conditions with API call helpers

GPS origin used for Blocks.exe default spawn (Redmond WA placeholder):
  lat=47.641468, lon=-122.140165, alt=122.0 m MSL

Coordinate conversion:
  north_m = (lat  - origin_lat) * 111_320
  east_m  = (lon  - origin_lon) * 111_320 * cos(origin_lat_rad)
  ↔  inverse for NED->GPS
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

# ─── World origin (matches AirSim OriginGeopoint default) ─────────────────────
ORIGIN_LAT = 47.641468
ORIGIN_LON = -122.140165
ORIGIN_ALT = 122.0          # metres MSL

_LAT_M  = 111_320.0         # metres per degree latitude (approx)
_LON_M  = _LAT_M * math.cos(math.radians(ORIGIN_LAT))


# ─── Coordinate helpers ───────────────────────────────────────────────────────

def ned_to_gps(north_m: float, east_m: float, alt_m: float = 0.0) -> tuple[float, float, float]:
    """AirSim NED (metres) → (lat, lon, alt_msl)."""
    lat = ORIGIN_LAT + north_m / _LAT_M
    lon = ORIGIN_LON + east_m  / _LON_M
    return lat, lon, ORIGIN_ALT + alt_m


def gps_to_ned(lat: float, lon: float, alt_msl: float = ORIGIN_ALT) -> tuple[float, float, float]:
    """(lat, lon, alt_msl) → AirSim NED (metres)."""
    north = (lat - ORIGIN_LAT) * _LAT_M
    east  = (lon - ORIGIN_LON) * _LON_M
    return north, east, alt_msl - ORIGIN_ALT


# ─── Static obstacle catalogue ────────────────────────────────────────────────

@dataclass
class Obstacle:
    name:    str
    north_m: float          # AirSim NED origin
    east_m:  float
    height_m: float         # approximate height above ground
    radius_m: float         # horizontal footprint radius

    @property
    def gps(self) -> tuple[float, float, float]:
        return ned_to_gps(self.north_m, self.east_m, self.height_m / 2)

    @property
    def clearance_alt(self) -> float:
        """Minimum safe flight altitude over this obstacle (m AGL)."""
        return self.height_m + 3.0


# Blocks.exe world has open flat terrain with a few large cubes.
# These positions match the default Blocks asset placements (approximate).
WORLD_OBSTACLES: list[Obstacle] = [
    Obstacle("building_A",  north_m= 20,  east_m=  10, height_m=15, radius_m=5),
    Obstacle("building_B",  north_m= 20,  east_m= -10, height_m=15, radius_m=5),
    Obstacle("tower_C",     north_m= 50,  east_m=   0, height_m=30, radius_m=4),
    Obstacle("wall_D",      north_m= 35,  east_m=  20, height_m= 8, radius_m=8),
    Obstacle("tree_cluster",north_m=-10,  east_m=  15, height_m= 6, radius_m=6),
    Obstacle("ground_box",  north_m=  5,  east_m= -20, height_m= 3, radius_m=3),
]

DRONE_STATION_1 = ned_to_gps(  0,   0, 0)   # takeoff pad near origin
DRONE_STATION_2 = ned_to_gps(  0, -30, 0)   # second pad 30 m west


# ─── Victim trajectory ────────────────────────────────────────────────────────

@dataclass
class Waypoint:
    lat:   float
    lon:   float
    alt_m: float = 0.0      # ground level by default
    speed_ms: float = 1.2   # walking speed m/s


@dataclass
class VictimTrajectory:
    """
    Deterministic + noise GPS path for a simulated victim.

    Trajectory types
    ----------------
    "walking"  : steady straight-line walk
    "panicked" : erratic direction changes (being chased)
    "stationary": victim collapses / frozen in place
    "circling" : victim pacing in distress loop

    Usage
    -----
    traj = VictimTrajectory("panicked", seed=42)
    for wp in traj.waypoints:
        drone.set_waypoint("DRONE_1", wp.lat, wp.lon, 30.0)
    """
    trajectory_type: str = "walking"
    seed: int = 42
    n_waypoints: int = 20
    start_north_m: float = 0.0
    start_east_m:  float = 0.0
    noise_sigma_m: float = 1.0   # positional noise per step

    waypoints: list[Waypoint] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        self.waypoints = self._generate(rng)

    def _generate(self, rng: random.Random) -> list[Waypoint]:
        n  = self.start_north_m
        e  = self.start_east_m
        wps: list[Waypoint] = []

        if self.trajectory_type == "walking":
            heading = 0.0              # heading in degrees (north)
            step_m  = 2.0
            for _ in range(self.n_waypoints):
                n += step_m * math.cos(math.radians(heading)) + rng.gauss(0, self.noise_sigma_m)
                e += step_m * math.sin(math.radians(heading)) + rng.gauss(0, self.noise_sigma_m)
                lat, lon, _ = ned_to_gps(n, e)
                wps.append(Waypoint(lat, lon, speed_ms=1.2))

        elif self.trajectory_type == "panicked":
            step_m = 3.5              # faster, irregular steps
            for i in range(self.n_waypoints):
                # sharp heading changes every 3-4 steps
                heading = rng.uniform(0, 360)
                n += step_m * math.cos(math.radians(heading)) + rng.gauss(0, self.noise_sigma_m * 2)
                e += step_m * math.sin(math.radians(heading)) + rng.gauss(0, self.noise_sigma_m * 2)
                lat, lon, _ = ned_to_gps(n, e)
                wps.append(Waypoint(lat, lon, speed_ms=3.5))

        elif self.trajectory_type == "stationary":
            lat, lon, _ = ned_to_gps(n, e)
            for _ in range(self.n_waypoints):
                wps.append(Waypoint(
                    lat + rng.gauss(0, 0.000001),
                    lon + rng.gauss(0, 0.000001),
                    speed_ms=0.0,
                ))

        elif self.trajectory_type == "circling":
            radius_m = 5.0
            for i in range(self.n_waypoints):
                angle = 2 * math.pi * i / self.n_waypoints
                cn = n + radius_m * math.cos(angle) + rng.gauss(0, self.noise_sigma_m)
                ce = e + radius_m * math.sin(angle) + rng.gauss(0, self.noise_sigma_m)
                lat, lon, _ = ned_to_gps(cn, ce)
                wps.append(Waypoint(lat, lon, speed_ms=0.8))

        else:
            raise ValueError(f"Unknown trajectory type: {self.trajectory_type!r}")

        return wps

    def bbox_ned(self) -> tuple[float, float, float, float]:
        """(min_north, max_north, min_east, max_east) of trajectory in NED metres."""
        nords = [gps_to_ned(wp.lat, wp.lon)[0] for wp in self.waypoints]
        easts = [gps_to_ned(wp.lat, wp.lon)[1] for wp in self.waypoints]
        return min(nords), max(nords), min(easts), max(easts)


# ─── Weather presets ──────────────────────────────────────────────────────────

class Weather(Enum):
    CLEAR      = "clear"
    FOG        = "fog"
    RAIN       = "rain"
    DUST       = "dust"
    NIGHT      = "night"


@dataclass
class WeatherConfig:
    name:        str
    fog_density: float = 0.0    # 0.0 – 1.0 (AirSim WeatherParameter.Fog)
    rain_amount: float = 0.0    # 0.0 – 1.0 (WeatherParameter.Rain)
    dust_amount: float = 0.0    # 0.0 – 1.0 (WeatherParameter.Dust)
    wind_speed:  float = 0.0    # m/s
    sun_angle:   float = 45.0   # degrees above horizon; 0 = night


WEATHER_PRESETS: dict[Weather, WeatherConfig] = {
    Weather.CLEAR: WeatherConfig("clear",      fog_density=0.0, rain_amount=0.0, sun_angle=60.0),
    Weather.FOG:   WeatherConfig("fog",        fog_density=0.7, rain_amount=0.0, sun_angle=45.0),
    Weather.RAIN:  WeatherConfig("rain",       fog_density=0.2, rain_amount=0.8, sun_angle=30.0),
    Weather.DUST:  WeatherConfig("dust",       fog_density=0.0, rain_amount=0.0, dust_amount=0.6, wind_speed=5.0, sun_angle=45.0),
    Weather.NIGHT: WeatherConfig("night",      fog_density=0.0, rain_amount=0.0, sun_angle=0.0),
}


def apply_weather(client, preset: Weather) -> None:
    """
    Apply a weather preset to AirSim via the Python API client.

    client  — airsim.MultirotorClient() (already connected)
    preset  — Weather enum member

    Requires AirSim built with weather enabled (Blocks.exe supports this).
    """
    try:
        import airsim
    except ImportError:
        print("[environment] airsim package not installed — weather not applied.")
        return

    cfg = WEATHER_PRESETS[preset]
    client.simEnableWeather(True)

    weather_map = {
        "fog_density": airsim.WeatherParameter.Fog,
        "rain_amount": airsim.WeatherParameter.Rain,
        "dust_amount": airsim.WeatherParameter.Dust,
    }
    for attr, param in weather_map.items():
        val = getattr(cfg, attr)
        if val > 0:
            client.simSetWeatherParamScalar(param, val)

    print(f"[environment] Weather set to {cfg.name!r}")


# ─── Zone helpers ─────────────────────────────────────────────────────────────

def safe_flight_altitude(obstacles: Sequence[Obstacle]) -> float:
    """Minimum altitude (m AGL) that clears all obstacles in the list."""
    return max((o.clearance_alt for o in obstacles), default=10.0)


def is_path_obstructed(
    start_ned: tuple[float, float],
    end_ned:   tuple[float, float],
    obstacles: Sequence[Obstacle],
    drone_alt_m: float,
) -> list[Obstacle]:
    """
    Return list of obstacles that intersect the straight 2-D line between
    start_ned and end_ned when drone flies at drone_alt_m AGL.

    Only checks obstacles taller than drone_alt_m.
    """
    n0, e0 = start_ned
    n1, e1 = end_ned
    blocking: list[Obstacle] = []

    for obs in obstacles:
        if drone_alt_m >= obs.clearance_alt:
            continue
        # Closest point on segment to obstacle centre
        dx, dy = n1 - n0, e1 - e0
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0:
            dist = math.hypot(obs.north_m - n0, obs.east_m - e0)
        else:
            t = max(0.0, min(1.0, ((obs.north_m - n0) * dx + (obs.east_m - e0) * dy) / seg_len_sq))
            cn = n0 + t * dx
            ce = e0 + t * dy
            dist = math.hypot(obs.north_m - cn, obs.east_m - ce)

        if dist < obs.radius_m:
            blocking.append(obs)

    return blocking


# ─── Quick sanity test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Environment module — sanity check")
    print(f"  Origin: {ORIGIN_LAT}, {ORIGIN_LON}, {ORIGIN_ALT}m")
    print(f"  Station 1: {DRONE_STATION_1}")
    print(f"  Station 2: {DRONE_STATION_2}")
    print(f"\n  Obstacles ({len(WORLD_OBSTACLES)}):")
    for obs in WORLD_OBSTACLES:
        lat, lon, _ = obs.gps
        print(f"    {obs.name:<16} lat={lat:.6f}  lon={lon:.6f}  h={obs.height_m}m  r={obs.radius_m}m  clearance={obs.clearance_alt}m")

    print("\n  Victim trajectories:")
    for ttype in ("walking", "panicked", "stationary", "circling"):
        traj = VictimTrajectory(ttype, n_waypoints=5)
        print(f"    {ttype:<12}: {len(traj.waypoints)} waypoints, "
              f"first=({traj.waypoints[0].lat:.6f}, {traj.waypoints[0].lon:.6f}), "
              f"speed={traj.waypoints[0].speed_ms} m/s")

    print("\n  Safe flight altitude (all obstacles):", safe_flight_altitude(WORLD_OBSTACLES), "m")

    start = (0.0, 0.0)
    end   = (60.0, 0.0)
    blocking = is_path_obstructed(start, end, WORLD_OBSTACLES, drone_alt_m=10.0)
    print(f"\n  Path (0,0)->(60,0) at 10m AGL: {[o.name for o in blocking]} blocked")
    blocking2 = is_path_obstructed(start, end, WORLD_OBSTACLES, drone_alt_m=35.0)
    print(f"  Path (0,0)->(60,0) at 35m AGL: {[o.name for o in blocking2]} blocked")

    print("\nOK")
