"""
PX4 OFFBOARD flight via pure pymavlink (no DroneKit).

DroneKit's HEARTBEAT handler raises APIException on PX4 custom modes
(OFFBOARD=393216) because DroneKit was designed for ArduPilot.
We use pymavlink directly and track state in a background receiver thread.

PX4 custom mode encoding (px4_custom_mode.h):
  main_mode in bits 16-23 → OFFBOARD = 6 << 16 = 393216

Run: python simulation/fly_test.py
Requires: AirSim (Blocks.exe) + PX4 SITL showing 'Ready for takeoff!'
"""
import sys, time, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymavlink import mavutil

CONNECT_STR  = "udp:127.0.0.1:14550"
TARGET_ALT   = 5.0          # metres AGL
PX4_OFFBOARD = 6 << 16      # = 393216 — OFFBOARD main_mode packed into bits 16-23

# ── Shared telemetry state (updated by receiver thread) ──────────────────────
_state = {"alt": 0.0, "armed": False, "custom_mode": 0}
_state_lock = threading.Lock()
_stop = False

# ── Connect ───────────────────────────────────────────────────────────────────
print("\nGuardian Drone — PX4 OFFBOARD Flight Demo")
print("=" * 44)
print(f"Connecting to {CONNECT_STR} ...")
mav = mavutil.mavlink_connection(CONNECT_STR)
mav.wait_heartbeat()
print(f"Heartbeat received  sys={mav.target_system}  comp={mav.target_component}")

# ── Background receiver: keeps _state up to date ──────────────────────────────
def _recv_loop():
    while not _stop:
        msg = mav.recv_match(blocking=True, timeout=0.1)
        if msg is None:
            continue
        t = msg.get_type()
        with _state_lock:
            if t == "HEARTBEAT":
                _state["armed"]       = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                _state["custom_mode"] = msg.custom_mode
            elif t == "GLOBAL_POSITION_INT":
                _state["alt"] = msg.relative_alt / 1000.0   # mm → m

threading.Thread(target=_recv_loop, daemon=True).start()

# ── Setpoint streamer (20 Hz) — start before OFFBOARD switch ─────────────────
_target_ned_z = 0.0   # z=0 = stay on ground; negative = up

def _stream_loop():
    while not _stop:
        mav.mav.set_position_target_local_ned_send(
            0,
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000,      # position only (ignore vel/accel/yaw)
            0.0, 0.0, _target_ned_z,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        time.sleep(0.05)             # 20 Hz

# ── Wait for EKF ──────────────────────────────────────────────────────────────
print("Waiting 25s for EKF to stabilise ...", flush=True)
for i in range(25, 0, -1):
    print(f"  {i:2d}s ...", end="\r", flush=True)
    time.sleep(1)
print()

# Start streaming before mode switch
threading.Thread(target=_stream_loop, daemon=True).start()
print("Setpoints streaming at 20 Hz. Waiting 3s before OFFBOARD ...")
time.sleep(3)

# ── Switch to OFFBOARD ────────────────────────────────────────────────────────
print("Switching to OFFBOARD (custom_mode=393216) ...")
t0 = time.time()
while True:
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        PX4_OFFBOARD,
    )
    time.sleep(0.5)
    with _state_lock:
        mode = _state["custom_mode"]
    if mode == PX4_OFFBOARD:
        print("OFFBOARD mode confirmed!")
        break
    if time.time() - t0 > 15:
        print(f"Timeout — custom_mode stuck at {mode}")
        _stop = True; sys.exit(1)

# ── Set climb target BEFORE arming (PX4 auto-disarms if z=0 at arm time) ─────
# PX4 sees z=-TARGET_ALT immediately on arm → knows flight is intended → stays armed
print(f"Pre-setting climb target to {TARGET_ALT}m ...")
_target_ned_z = -TARGET_ALT    # NED: negative z = up
time.sleep(0.3)                # let stream thread send a few setpoints with new target

# ── Arm (force) ───────────────────────────────────────────────────────────────
print("Arming (force) ...")
mav.mav.command_long_send(
    mav.target_system, mav.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 1, 21196, 0, 0, 0, 0, 0,
)
t0 = time.time()
while True:
    with _state_lock:
        armed = _state["armed"]
    if armed:
        print("Armed!")
        break
    if time.time() - t0 > 10:
        print("Arming timeout — check PX4 terminal for ARMING_CHECK failures")
        _stop = True; sys.exit(1)
    time.sleep(0.3)

# ── Climb to TARGET_ALT ───────────────────────────────────────────────────────
print(f"Climbing to {TARGET_ALT}m AGL ...")

t0 = time.time()
while True:
    with _state_lock:
        alt   = _state["alt"]
        armed = _state["armed"]
    print(f"  Alt: {alt:5.1f}m  armed={armed}", end="\r")
    if alt >= TARGET_ALT * 0.85:
        print(f"\nReached {alt:.1f}m!")
        break
    if not armed:
        print(f"\nDisarmed unexpectedly at {alt:.1f}m — check PX4 terminal")
        _stop = True; sys.exit(1)
    if time.time() - t0 > 30:
        print(f"\nTimeout — alt={alt:.1f}m  (expected >={TARGET_ALT * 0.85:.1f}m)")
        break
    time.sleep(0.5)

# ── Hover 5 s ─────────────────────────────────────────────────────────────────
print("Hovering 5 s ...")
time.sleep(5)

# ── Descend ───────────────────────────────────────────────────────────────────
print("Descending ...")
_target_ned_z = 0.0
t0 = time.time()
while True:
    with _state_lock:
        alt = _state["alt"]
    print(f"  Alt: {alt:5.1f}m", end="\r")
    if alt < 0.5 or time.time() - t0 > 20:
        break
    time.sleep(0.5)

with _state_lock:
    final_alt = _state["alt"]
print(f"\nLanded. Final alt = {final_alt:.1f}m")
_stop = True
time.sleep(0.5)
print("Done.")
