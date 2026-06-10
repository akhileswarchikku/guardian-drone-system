"""
Attack scene manager for Guardian Drone integration demo.

Creates a simulated street assault:
  - Victim (woman) represented as a blue sphere in AirSim
  - Attacker (man)  represented as a red  sphere in AirSim

Both actors move continuously:
  - Victim  : flees in a semi-random direction (panicked movement)
  - Attacker: steers toward victim (chasing)

GPS positions are tracked in real Hyderabad coordinates so the agent
graph and database receive realistic location data.
NED positions are derived from those GPS coords so AirSim visuals match.
"""
from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass

from simulation.airsim_client import AirSimClient
from simulation.environment import ORIGIN_LAT, ORIGIN_LON, gps_to_ned


# ── Starting positions (~60 miles from Hyderabad, near Yadadri district) ────────
# Distance from Madhapur PS (17.4410N, 78.3830E): 97.9 km = 60.8 miles
VICTIM_LAT   = 17.8100
VICTIM_LON   = 79.2200
ATTACKER_LAT = 17.80987   # ~21 m SW of victim — attacker right on top of victim
ATTACKER_LON = 79.21986


@dataclass
class _Actor:
    lat: float
    lon: float
    speed_ms: float
    heading_deg: float = 0.0


class AttackScene:
    """
    Manages victim + attacker GPS positions and AirSim visual actors.

    Usage:
        scene = AttackScene(airsim_client)
        scene.setup()   # spawn spheres in AirSim
        scene.start()   # begin movement loop in background thread
        lat, lon = scene.victim_gps()
        scene.stop()
    """

    def __init__(self, airsim: AirSimClient) -> None:
        self._airsim = airsim
        self._victim   = _Actor(VICTIM_LAT,   VICTIM_LON,   speed_ms=2.0,
                                heading_deg=random.uniform(30, 90))
        self._attacker = _Actor(ATTACKER_LAT, ATTACKER_LON, speed_ms=1.9,
                                heading_deg=0.0)
        self._lock    = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Print initial positions; optionally spawn spheres in AirSim."""
        print(f"  [Scene] Victim    (blue) @ {VICTIM_LAT:.5f}N  {VICTIM_LON:.5f}E")
        print(f"  [Scene] Attacker  (red)  @ {ATTACKER_LAT:.5f}N  {ATTACKER_LON:.5f}E")

        # Convert to AirSim NED (Blocks world origin ≠ Hyderabad, so offset only)
        vn, ve = self._to_local_ned(VICTIM_LAT, VICTIM_LON)
        an, ae = self._to_local_ned(ATTACKER_LAT, ATTACKER_LON)

        ok_v = self._airsim.spawn_sphere("Victim",   vn, ve, z=-0.5, scale=1.5)
        ok_a = self._airsim.spawn_sphere("Attacker", an, ae, z=-0.5, scale=1.5)

        if ok_v and ok_a:
            print("  [Scene] Actors spawned in AirSim Blocks world")
        else:
            print("  [Scene] AirSim actors unavailable — GPS tracking still active")

    def start(self) -> None:
        """Start background thread that moves both actors every 0.5 s."""
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def victim_gps(self) -> tuple[float, float]:
        with self._lock:
            return self._victim.lat, self._victim.lon

    def attacker_gps(self) -> tuple[float, float]:
        with self._lock:
            return self._attacker.lat, self._attacker.lon

    def distance_m(self) -> float:
        """Current distance between victim and attacker in metres."""
        with self._lock:
            return self._haversine_m(
                self._victim.lat, self._victim.lon,
                self._attacker.lat, self._attacker.lon,
            )

    # ── Movement loop (background thread) ────────────────────────────────────

    def _loop(self) -> None:
        dt = 0.5
        while self._running:
            with self._lock:
                # Victim: flee, occasionally veer
                if random.random() < 0.2:
                    self._victim.heading_deg += random.uniform(-40, 40)
                self._step(self._victim, dt)

                # Attacker: chase victim
                bearing = self._bearing(
                    self._attacker.lat, self._attacker.lon,
                    self._victim.lat,   self._victim.lon,
                )
                self._attacker.heading_deg = bearing + random.gauss(0, 8)
                self._step(self._attacker, dt)

                vlat, vlon = self._victim.lat,   self._victim.lon
                alat, alon = self._attacker.lat, self._attacker.lon

            # Update AirSim visuals
            vn, ve = self._to_local_ned(vlat, vlon)
            an, ae = self._to_local_ned(alat, alon)
            self._airsim.move_actor("Victim",   vn, ve, -0.5)
            self._airsim.move_actor("Attacker", an, ae, -0.5)

            time.sleep(dt)

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _step(actor: _Actor, dt: float) -> None:
        """Advance actor by speed*dt metres in heading_deg direction."""
        dist_m  = actor.speed_ms * dt
        hdg_rad = math.radians(actor.heading_deg)
        dlat    = (dist_m * math.cos(hdg_rad)) / 111_320.0
        dlon    = (dist_m * math.sin(hdg_rad)) / (
            111_320.0 * math.cos(math.radians(actor.lat)) + 1e-9
        )
        actor.lat += dlat
        actor.lon += dlon

    @staticmethod
    def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compass bearing (degrees) from point 1 to point 2."""
        la1, lo1 = math.radians(lat1), math.radians(lon1)
        la2, lo2 = math.radians(lat2), math.radians(lon2)
        x = math.sin(lo2 - lo1) * math.cos(la2)
        y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1)
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    @staticmethod
    def _haversine_m(lat1, lon1, lat2, lon2) -> float:
        R = 6_371_000
        la1, la2 = math.radians(lat1), math.radians(lat2)
        dl = math.radians(lon2 - lon1)
        dp = math.radians(lat2 - lat1)
        a  = math.sin(dp/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dl/2)**2
        return 2 * R * math.asin(math.sqrt(a))

    @staticmethod
    def _to_local_ned(lat: float, lon: float) -> tuple[float, float]:
        """
        Convert GPS to NED offset relative to AirSim Blocks world origin.
        We reuse gps_to_ned() from environment.py (origin = Blocks spawn point).
        The z/altitude component is set by the caller.
        """
        north, east, _ = gps_to_ned(lat, lon, 0.0)
        # Clamp to AirSim Blocks world bounds (~500m radius from origin)
        north = max(-400.0, min(400.0, north))
        east  = max(-400.0, min(400.0, east))
        return north, east
