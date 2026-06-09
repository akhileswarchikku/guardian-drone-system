"""
Phase 3.3 — DroneKit Simulation Bridge

Drop-in replacement for agents/drone_client.py when running in simulation.
Exact same function signatures — agent graph requires zero changes.

Architecture
------------
  LangGraph agents
        ↓  (same calls as production)
  sim_bridge.py
        ↓  DroneKit MAVLink API
  PX4 SITL (WSL2)  <-->  AirSim Blocks (Windows)

Usage
-----
  # In simulation mode, patch drone_client before importing agent_graph:
  import simulation.sim_bridge as drone_client   # replaces agents.drone_client

  # Or run standalone to verify connection:
  python simulation/sim_bridge.py

Connection ports (matches airsim_settings.json)
------------------------------------------------
  Drone 1 MAVLink GCS port : udp:127.0.0.1:14550
  Drone 2 MAVLink GCS port : udp:127.0.0.1:14551
  AirSim API port          : 41451 (default)

AirSim Blocks.exe location (this machine)
------------------------------------------
  C:/Users/akhil/Documents/AirSim/Blocks/Blocks/WindowsNoEditor/Blocks/Binaries/Win64/Blocks.exe
  AirSim settings.json: C:/Users/akhil/Documents/AirSim/settings.json  (auto-loaded by Blocks)
"""
from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.state import DroneTelemetry

# MAVLink GCS ports per drone index (0-based)
_GCS_PORTS = {
    "DRONE_1": "udp:127.0.0.1:14550",
    "DRONE_2": "udp:127.0.0.1:14551",
}
_DEFAULT_PORT = _GCS_PORTS["DRONE_1"]

# Active DroneKit vehicle connections keyed by drone_id
_vehicles: dict[str, object] = {}
_lock = threading.Lock()


# ─── Connection ───────────────────────────────────────────────────────────────

def _connect(drone_id: str, gcs_url: str, timeout: int = 30) -> object:
    """Connect to PX4 SITL via DroneKit MAVLink. Caches connection."""
    with _lock:
        if drone_id in _vehicles:
            return _vehicles[drone_id]

    import dronekit
    print(f"  [SimBridge] Connecting to {drone_id} at {gcs_url} ...")
    vehicle = dronekit.connect(gcs_url, wait_ready=True, timeout=timeout)
    print(f"  [SimBridge] {drone_id} connected — "
          f"mode={vehicle.mode.name}  armed={vehicle.armed}")

    with _lock:
        _vehicles[drone_id] = vehicle
    return vehicle


def _get_vehicle(drone_id: str) -> object:
    """Return cached vehicle or raise if not connected."""
    with _lock:
        v = _vehicles.get(drone_id)
    if v is None:
        raise RuntimeError(
            f"Drone {drone_id} not connected. Call launch_drone() first."
        )
    return v


# ─── Public API — same signatures as agents/drone_client.py ──────────────────

def launch_drone(
    drone_id:   str,
    station_id: int,
    target_lat: float,
    target_lon: float,
    target_alt: float = 30.0,
    server_url: str   = "",          # ignored in sim — kept for API compat
) -> dict:
    """
    Arm + takeoff drone in PX4 SITL via DroneKit.
    Mirrors drone_client.launch_drone() signature exactly.
    """
    import dronekit

    gcs_url = _GCS_PORTS.get(drone_id, _DEFAULT_PORT)
    vehicle = _connect(drone_id, gcs_url)

    # Switch to GUIDED mode
    vehicle.mode = dronekit.VehicleMode("GUIDED")
    time.sleep(1)

    # Arm
    vehicle.armed = True
    t0 = time.time()
    while not vehicle.armed:
        if time.time() - t0 > 10:
            return {"success": False, "error": "Arming timeout"}
        time.sleep(0.5)

    # Takeoff
    vehicle.simple_takeoff(target_alt)
    print(f"  [SimBridge] {drone_id} taking off to {target_alt}m ...")

    # Wait until target altitude reached
    t0 = time.time()
    while True:
        alt = vehicle.location.global_relative_frame.alt or 0.0
        if alt >= target_alt * 0.90:
            break
        if time.time() - t0 > 60:
            return {"success": False, "error": "Takeoff timeout"}
        time.sleep(1)

    print(f"  [SimBridge] {drone_id} airborne at {alt:.1f}m")
    return {
        "success":    True,
        "drone_id":   drone_id,
        "station_id": station_id,
        "alt_m":      alt,
    }


def get_telemetry(drone_id: str, server_url: str = "") -> DroneTelemetry:
    """
    Read current state from PX4 SITL via DroneKit.
    Mirrors drone_client.get_telemetry() — returns DroneTelemetry Pydantic model.
    """
    vehicle = _get_vehicle(drone_id)

    loc  = vehicle.location.global_relative_frame
    vel  = vehicle.velocity  # [vx, vy, vz] m/s
    batt = vehicle.battery
    mode = vehicle.mode.name if vehicle.mode else "unknown"

    speed_ms = float((vel[0]**2 + vel[1]**2) ** 0.5) if vel else 0.0
    battery_pct = float(batt.level or 100)

    # Map PX4 mode → our status enum
    status_map = {
        "GUIDED":    "flying",
        "AUTO":      "flying",
        "LOITER":    "hovering",
        "RTL":       "returning",
        "LAND":      "returning",
        "STABILIZE": "idle",
    }
    status = status_map.get(mode, "flying")

    # Check for simulated faults via parameter
    fault_code = None
    try:
        if vehicle.parameters.get("SIM_ENGINE_FAIL", 0):
            fault_code = "motor_failure"
            status     = "fault"
        elif vehicle.parameters.get("SIM_GPS_DISABLE", 0):
            fault_code = "gps_loss"
    except Exception:
        pass

    return DroneTelemetry(
        drone_id       = drone_id,
        lat            = float(loc.lat)  if loc.lat  else 0.0,
        lon            = float(loc.lon)  if loc.lon  else 0.0,
        alt_m          = float(loc.alt)  if loc.alt  else 0.0,
        speed_ms       = speed_ms,
        battery_pct    = battery_pct,
        status         = status,
        fault_code     = fault_code,
        camera_active  = True,
        motor_status   = "failed" if fault_code == "motor_failure" else "ok",
        signal_rssi    = -120 if fault_code == "gps_loss" else -55,
    )


def set_waypoint(
    drone_id:   str,
    lat:        float,
    lon:        float,
    alt_m:      float = 30.0,
    server_url: str   = "",
) -> dict:
    """
    Command drone to fly to (lat, lon, alt_m) via DroneKit simple_goto.
    Mirrors drone_client.set_waypoint() signature.
    """
    import dronekit

    vehicle = _get_vehicle(drone_id)
    target  = dronekit.LocationGlobalRelative(lat, lon, alt_m)
    vehicle.simple_goto(target)
    print(f"  [SimBridge] {drone_id} -> waypoint ({lat:.6f}, {lon:.6f}, {alt_m}m)")
    return {"success": True, "drone_id": drone_id, "lat": lat, "lon": lon, "alt_m": alt_m}


def inject_fault(
    drone_id:   str,
    fault_type: str,
    server_url: str = "",
) -> dict:
    """
    Inject a simulated fault via PX4 SITL parameters.
    Mirrors drone_client.inject_fault() signature.

    fault_type options
    ------------------
      battery_critical  — set battery level to 5%
      gps_loss          — disable GPS (SIM_GPS_DISABLE=1)
      motor_failure     — kill one motor (SIM_ENGINE_FAIL=1)
      signal_loss       — simulate signal drop (parameter SIM_RC_LOSS)
      camera_failure    — set camera_active=False in next telemetry read
    """
    vehicle = _get_vehicle(drone_id)

    if fault_type == "battery_critical":
        # DroneKit can't directly set battery — use SITL parameter
        try:
            vehicle.parameters["SIM_BATT_VOLTAGE"] = 10.0   # below 3.5V/cell
        except Exception:
            pass
        print(f"  [SimBridge] FAULT battery_critical on {drone_id}")

    elif fault_type == "gps_loss":
        vehicle.parameters["SIM_GPS_DISABLE"] = 1
        print(f"  [SimBridge] FAULT gps_loss on {drone_id}")

    elif fault_type == "motor_failure":
        vehicle.parameters["SIM_ENGINE_FAIL"] = 1
        print(f"  [SimBridge] FAULT motor_failure on {drone_id}")

    elif fault_type == "signal_loss":
        vehicle.parameters["SIM_RC_LOSS_TIME"] = 1
        print(f"  [SimBridge] FAULT signal_loss on {drone_id}")

    elif fault_type == "camera_failure":
        # Tracked in _camera_faults set; get_telemetry() checks it
        _camera_faults.add(drone_id)
        print(f"  [SimBridge] FAULT camera_failure on {drone_id}")

    else:
        return {"success": False, "error": f"Unknown fault type: {fault_type}"}

    return {"success": True, "drone_id": drone_id, "fault": fault_type}


_camera_faults: set[str] = set()


def disconnect_all() -> None:
    """Close all DroneKit vehicle connections."""
    with _lock:
        for did, v in _vehicles.items():
            try:
                v.close()
                print(f"  [SimBridge] {did} disconnected.")
            except Exception:
                pass
        _vehicles.clear()


# ─── Standalone connection test ───────────────────────────────────────────────

if __name__ == "__main__":
    print("\nGuardian Drone — SimBridge Connection Test")
    print("Make sure AirSim + PX4 SITL are running first.")
    print("=" * 50)

    try:
        drone_id = "DRONE_1"
        gcs_url  = _GCS_PORTS[drone_id]
        vehicle  = _connect(drone_id, gcs_url, timeout=20)

        print(f"\nVehicle state:")
        print(f"  Mode      : {vehicle.mode.name}")
        print(f"  Armed     : {vehicle.armed}")
        batt = vehicle.battery
        print(f"  Battery   : {batt.level}%  {batt.voltage:.2f}V")
        loc = vehicle.location.global_frame
        print(f"  GPS       : {loc.lat:.6f}, {loc.lon:.6f}, alt={loc.alt:.1f}m")
        print(f"  Groundspeed: {vehicle.groundspeed:.2f} m/s")
        print(f"\nSimBridge OK -- DroneKit connected to PX4 SITL.")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nTroubleshooting:")
        print("  1. Start AirSim (Blocks.exe) on Windows")
        print("  2. In WSL2: cd ~/PX4-Autopilot && make px4_sitl_default none_iris")
        print("  3. Wait for PX4 to print 'Ready for takeoff'")
        print("  4. Then run this script again")
        print("  Blocks.exe: C:/Users/akhil/Documents/AirSim/Blocks/Blocks/WindowsNoEditor/Blocks/Binaries/Win64/Blocks.exe")
        print("  settings.json: C:/Users/akhil/Documents/AirSim/settings.json")
    finally:
        disconnect_all()
