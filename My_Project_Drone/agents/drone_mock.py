"""
Phase 2 — Mock Drone Server (FastAPI)

Simulates police drone telemetry for testing without real hardware.
Runs on port 8001 by default.

Run standalone:
    C:/Users/akhil/anaconda3/envs/LLM_GPU/python.exe agents/drone_mock.py

API:
    POST /drone/{id}/launch       — start mission from home position
    GET  /drone/{id}/telemetry    — get current state (moves drone each call)
    POST /drone/{id}/waypoint     — update target
    POST /drone/{id}/land         — land in place
    POST /drone/{id}/inject_fault — inject fault for malfunction testing
    DELETE /drone/{id}            — reset to idle (test teardown)
    GET  /health                  — liveness check
"""
from __future__ import annotations

import math
import sys
import threading
from pathlib import Path
from typing import Literal, Optional

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Guardian Drone Mock", version="2.0")

# ── In-memory drone registry ──────────────────────────────────────────────────
_lock   = threading.Lock()
_drones: dict[str, "_DroneState"] = {}


class _DroneState:
    def __init__(self, drone_id: str, home_lat: float, home_lon: float):
        self.drone_id   = drone_id
        self.lat        = home_lat
        self.lon        = home_lon
        self.home_lat   = home_lat
        self.home_lon   = home_lon
        self.alt_m      = 0.0
        self.target_lat = home_lat
        self.target_lon = home_lon
        self.target_alt = 0.0
        self.speed_ms   = 0.0
        self.battery    = 100.0
        self.status: Literal["idle","flying","hovering","returning","fault"] = "idle"
        self.fault_code: Optional[str] = None
        self.step       = 0   # telemetry call counter
        # Extended telemetry (spec Phase 2 gap-fix)
        self.camera_active: bool = False
        self.motor_status: str   = "ok"   # ok | warning | failed
        self.signal_rssi: int    = -55    # dBm; -120 = lost

    def _dist_to_target_km(self) -> float:
        return _haversine_km(self.lat, self.lon, self.target_lat, self.target_lon)

    def advance(self) -> None:
        """Called on each GET /telemetry: move toward target, drain battery."""
        self.step += 1
        self.battery = max(0.0, self.battery - 0.3)   # 0.3% per poll

        if self.status == "fault":
            # Update derived fields for active fault type
            if self.fault_code == "motor_failure":
                self.motor_status  = "failed"
                self.camera_active = False
            elif self.fault_code == "gps_loss":
                self.signal_rssi = -120
            return

        if self.status in ("idle", "hovering"):
            return

        dist = self._dist_to_target_km()
        if dist < 0.05:
            self.lat    = self.target_lat
            self.lon    = self.target_lon
            self.alt_m  = self.target_alt
            self.status = "hovering"
            self.speed_ms = 0.0
            self.camera_active = True   # camera on while hovering at target
            return

        # Move 25% of remaining distance each poll (geometric convergence)
        frac = 0.25
        self.lat  += frac * (self.target_lat - self.lat)
        self.lon  += frac * (self.target_lon - self.lon)
        self.alt_m = max(self.alt_m + 5.0, self.target_alt)
        self.alt_m = min(self.alt_m, self.target_alt)
        self.speed_ms  = 15.0
        self.camera_active = True       # camera active during flight

        if self.battery <= 0:
            self.status    = "fault"
            self.fault_code = "battery_empty"
            self.camera_active = False

    def to_dict(self) -> dict:
        return {
            "drone_id":     self.drone_id,
            "lat":          round(self.lat, 7),
            "lon":          round(self.lon, 7),
            "alt_m":        round(self.alt_m, 1),
            "speed_ms":     self.speed_ms,
            "battery_pct":  round(self.battery, 1),
            "status":       self.status,
            "fault_code":   self.fault_code,
            "camera_active": self.camera_active,
            "motor_status": self.motor_status,
            "signal_rssi":  self.signal_rssi,
        }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _get_or_create(drone_id: str, home_lat: float = 17.44, home_lon: float = 78.39) -> _DroneState:
    if drone_id not in _drones:
        _drones[drone_id] = _DroneState(drone_id, home_lat, home_lon)
    return _drones[drone_id]


# ── Request / Response models ─────────────────────────────────────────────────

class LaunchRequest(BaseModel):
    home_lat:  float
    home_lon:  float
    waypoints: list[dict]   # list of {"lat","lon","alt_m"}


class WaypointRequest(BaseModel):
    lat:   float
    lon:   float
    alt_m: float = 30.0


class FaultRequest(BaseModel):
    fault_type: Literal["battery_critical", "motor_failure", "gps_loss"]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "drones_registered": len(_drones)}


@app.post("/drone/{drone_id}/launch")
def launch(drone_id: str, req: LaunchRequest):
    with _lock:
        d = _DroneState(drone_id, req.home_lat, req.home_lon)
        _drones[drone_id] = d
        if req.waypoints:
            final = req.waypoints[-1]
            d.target_lat = final["lat"]
            d.target_lon = final["lon"]
            d.target_alt = final.get("alt_m", 30.0)
        d.status   = "flying"
        d.alt_m    = 0.0
        d.speed_ms = 15.0
    return {"status": "launched", "drone_id": drone_id}


@app.get("/drone/{drone_id}/telemetry")
def telemetry(drone_id: str):
    with _lock:
        if drone_id not in _drones:
            raise HTTPException(status_code=404, detail=f"Drone {drone_id} not found")
        d = _drones[drone_id]
        d.advance()
        return d.to_dict()


@app.post("/drone/{drone_id}/waypoint")
def set_waypoint(drone_id: str, req: WaypointRequest):
    with _lock:
        d = _get_or_create(drone_id)
        d.target_lat = req.lat
        d.target_lon = req.lon
        d.target_alt = req.alt_m
        if d.status == "hovering":
            d.status = "flying"
    return {"status": "waypoint_set", "drone_id": drone_id}


@app.post("/drone/{drone_id}/land")
def land(drone_id: str):
    with _lock:
        d = _get_or_create(drone_id)
        d.status    = "idle"
        d.speed_ms  = 0.0
        d.alt_m     = 0.0
        d.fault_code = None
    return {"status": "landed", "drone_id": drone_id}


@app.post("/drone/{drone_id}/inject_fault")
def inject_fault(drone_id: str, req: FaultRequest):
    with _lock:
        d = _get_or_create(drone_id)
        d.status    = "fault"
        d.fault_code = req.fault_type
        if req.fault_type == "battery_critical":
            d.battery = 5.0
        d.speed_ms = 0.0
    return {"status": "fault_injected", "drone_id": drone_id, "fault": req.fault_type}


@app.delete("/drone/{drone_id}")
def reset_drone(drone_id: str):
    with _lock:
        _drones.pop(drone_id, None)
    return {"status": "reset", "drone_id": drone_id}


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Guardian Drone Mock Server on http://localhost:8001")
    print("Press Ctrl+C to stop.")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
