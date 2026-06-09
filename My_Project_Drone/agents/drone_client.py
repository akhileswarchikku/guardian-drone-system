"""
Phase 2 — HTTP client for the mock drone server (agents/drone_mock.py).
Each function raises httpx.HTTPStatusError on non-2xx responses.
Pass server_url to target a different port for multi-drone tests.
"""
from __future__ import annotations

import httpx

DEFAULT_URL = "http://localhost:8001"
TIMEOUT     = 5.0


def launch_drone(
    drone_id:   str,
    home_lat:   float,
    home_lon:   float,
    waypoints:  list[dict],
    server_url: str = DEFAULT_URL,
) -> dict:
    r = httpx.post(
        f"{server_url}/drone/{drone_id}/launch",
        json={"home_lat": home_lat, "home_lon": home_lon, "waypoints": waypoints},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def get_telemetry(drone_id: str, server_url: str = DEFAULT_URL) -> dict:
    r = httpx.get(f"{server_url}/drone/{drone_id}/telemetry", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def set_waypoint(
    drone_id:   str,
    lat:        float,
    lon:        float,
    alt_m:      float = 30.0,
    server_url: str   = DEFAULT_URL,
) -> dict:
    r = httpx.post(
        f"{server_url}/drone/{drone_id}/waypoint",
        json={"lat": lat, "lon": lon, "alt_m": alt_m},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def land_drone(drone_id: str, server_url: str = DEFAULT_URL) -> dict:
    r = httpx.post(f"{server_url}/drone/{drone_id}/land", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def inject_fault(
    drone_id:   str,
    fault_type: str,
    server_url: str = DEFAULT_URL,
) -> dict:
    """Inject a simulated fault: 'battery_critical', 'motor_failure', 'gps_loss'."""
    r = httpx.post(
        f"{server_url}/drone/{drone_id}/inject_fault",
        json={"fault_type": fault_type},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def reset_drone(drone_id: str, server_url: str = DEFAULT_URL) -> dict:
    """Reset a drone to idle state (for tests)."""
    r = httpx.delete(f"{server_url}/drone/{drone_id}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()
