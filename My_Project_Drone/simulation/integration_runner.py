"""
Phase 3.7 — Guardian Drone Full Integration Demo

Simulates a real street assault and exercises the complete AI stack:

  AirSim Blocks (visual scene)
    Victim (blue sphere) + Attacker (red sphere) moving in real-time

  Apple Watch biometric signal
    Baseline → attack spike → ContextualGate all-5-gates pass → SOS

  LangGraph 10-agent pipeline
    danger_score → sos_broadcast → station_finder → path_planner → dispatch

  PX4 SITL drone (AirSim physics)
    OFFBOARD mode → takeoff → fly to victim GPS → track moving victim

  Vision pipeline (YOLOv8 + MiDaS + scene classifier)
    AirSim camera frame (or synthetic) → detect threat → scene intelligence

Prerequisites
-------------
  1. WSL2: make px4_sitl_default none_iris
  2. Windows: launch AirSim Blocks.exe
  3. Wait for PX4: "Ready for takeoff!"
  4. Optionally: pip install airsim  (for visual actors + camera)
  5. Optionally: PostgreSQL running  (for real station DB)

Run
---
  conda activate LLM_GPU
  python simulation/integration_runner.py

  --no-fly   skip PX4 flight (agent graph only)
  --no-px4   same as --no-fly
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

# Windows terminals default to cp1252 — force UTF-8 so box/arrow chars print cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Banner ────────────────────────────────────────────────────────────────────

def _banner(msg: str, width: int = 60) -> None:
    bar = "=" * width
    print(f"\n  +{bar}+")
    for line in msg.strip().split("\n"):
        print(f"  |  {line:<{width - 2}}|")
    print(f"  +{bar}+\n")


# ── Phase helpers ─────────────────────────────────────────────────────────────

def _phase_header(label: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")


# ── Drone flight (pure pymavlink OFFBOARD) ────────────────────────────────────

def _fly_to_victim(
    ned_target: list,
    ned_lock: threading.Lock,
    stop_event: threading.Event,
    airsim_client=None,
    mock=None,
) -> None:
    """
    Fly DRONE_1 via PX4 OFFBOARD using the shared ned_target list [north, east, z].
    ned_target is the same list object that the GPS tracking loop updates in main(),
    so drone waypoint stays in sync with victim movement automatically.

    airsim_client + mock: passed in so the obstacle avoidance inner loop can grab
    camera/depth frames without creating a new connection.

    NOTE: AirSim Blocks world origin is Seattle; Hyderabad GPS → NED = 15M metres.
    ned_target must contain local AirSim coordinates (e.g. [50, 0, -25]).
    """
    from pymavlink import mavutil

    CONNECT_STR  = "udp:127.0.0.1:14550"
    PX4_OFFBOARD = 6 << 16       # custom_mode = 393216

    _state: dict = {"alt": 0.0, "armed": False, "custom_mode": 0}
    _lock  = threading.Lock()
    _stop_stream = threading.Event()
    _ned_target = ned_target   # shared with GPS tracking loop in main()

    try:
        print("  [Drone] Connecting to PX4 SITL (udp:127.0.0.1:14550) ...")
        mav = mavutil.mavlink_connection(CONNECT_STR)
        mav.wait_heartbeat(timeout=20)
        print(f"  [Drone] Heartbeat  sys={mav.target_system}  target={_ned_target[:2]}")
    except Exception as e:
        print(f"  [Drone] PX4 not reachable ({e}) — skipping flight")
        return

    # Background receiver
    def _recv():
        while not _stop_stream.is_set():
            msg = mav.recv_match(blocking=True, timeout=0.1)
            if not msg:
                continue
            with _lock:
                t = msg.get_type()
                if t == "HEARTBEAT":
                    _state["armed"]       = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    _state["custom_mode"] = msg.custom_mode
                elif t == "GLOBAL_POSITION_INT":
                    _state["alt"] = msg.relative_alt / 1000.0

    threading.Thread(target=_recv, daemon=True).start()

    # EKF warmup — must happen BEFORE stream starts (mirrors fly_test.py).
    # 25s gives PX4 time to acquire GPS lock and settle EKF.
    print("  [Drone] EKF warmup 25s (waiting for GPS lock + EKF settle) ...")
    for i in range(25, 0, -1):
        print(f"  [Drone]   {i:2d}s ...", end="\r")
        time.sleep(1)
    print()

    # Setpoint streamer — started AFTER EKF warmup, initially z=0 (ground hold).
    # z is updated to cruise altitude right before arming (same pattern as fly_test.py).
    _stream_z = [0.0]   # mutable so _stream closure can see updates

    def _stream():
        while not _stop_stream.is_set():
            with _lock:
                n, e = _ned_target[0], _ned_target[1]
                z    = _stream_z[0]
            mav.mav.set_position_target_local_ned_send(
                0, mav.target_system, mav.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000,
                n, e, z,
                0, 0, 0, 0, 0, 0, 0, 0,
            )
            time.sleep(0.05)

    threading.Thread(target=_stream, daemon=True).start()

    # Stream at z=0 for 3s so PX4 registers stable setpoints before OFFBOARD switch.
    print("  [Drone] Setpoints streaming (z=0, 3s warmup before OFFBOARD) ...")
    time.sleep(3)

    # Switch to OFFBOARD
    print("  [Drone] Switching to OFFBOARD ...")
    t0 = time.time()
    while True:
        mav.mav.set_mode_send(
            mav.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            PX4_OFFBOARD,
        )
        time.sleep(0.5)
        with _lock:
            mode = _state["custom_mode"]
        if mode == PX4_OFFBOARD:
            print("  [Drone] OFFBOARD confirmed")
            break
        if time.time() - t0 > 15:
            print("  [Drone] OFFBOARD timeout — aborting flight")
            _stop_stream.set()
            return

    # Set climb target BEFORE arming — PX4 sees z<0 immediately on arm → flight intent.
    # (z=0 at arm time → PX4 interprets as no takeoff intent → auto-disarms)
    cruise_alt = abs(_ned_target[2])   # ned_target[2] = -10.0 → cruise_alt = 10.0
    _stream_z[0] = -cruise_alt
    time.sleep(0.3)   # let stream send a few setpoints with climb target

    # Arm
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0,
    )
    t0 = time.time()
    while True:
        with _lock:
            armed = _state["armed"]
        if armed:
            print("  [Drone] Armed!")
            break
        if time.time() - t0 > 10:
            print("  [Drone] Arm timeout")
            _stop_stream.set()
            return
        time.sleep(0.3)

    # Climb
    # cruise_alt is already set above (before arm) — don't redeclare
    print(f"  [Drone] Climbing to {cruise_alt:.0f}m AGL ...")
    t0 = time.time()
    while True:
        with _lock:
            alt   = _state["alt"]
            armed = _state["armed"]
        print(f"  [Drone] alt={alt:5.1f}m", end="\r")
        if alt >= cruise_alt * 0.80:
            print(f"\n  [Drone] Reached {alt:.1f}m — en route to victim")
            break
        if not armed:
            print(f"\n  [Drone] Disarmed at {alt:.1f}m")
            _stop_stream.set()
            return
        if time.time() - t0 > 40:
            print(f"\n  [Drone] Climb timeout at {alt:.1f}m")
            break
        time.sleep(0.5)

    # ── Obstacle avoidance (A* + MiDaS depth) ────────────────────────────────
    # Inner closure: has direct access to _stream_z, ned_target, ned_lock,
    # stop_event, airsim_client, mock, cruise_alt — no extra plumbing needed.
    def _avoid_loop(interval_s: float = 5.0) -> None:
        try:
            from vision.obstacle_mapper import ObstacleMapper
            from vision.occupancy_grid import plan_path, assign_altitudes
        except ImportError as e:
            print(f"  [Avoid] Import failed: {e}")
            return

        mapper = ObstacleMapper(model_size="n", use_midas=True)
        print("  [Avoid] Obstacle avoidance active (YOLOv8-seg + MiDaS + A*, every 5s)")
        PX_TO_M = 0.10  # camera FOV 90°, ~0.1 m/px lateral at typical range

        while not stop_event.is_set():
            time.sleep(interval_s)
            if stop_event.is_set():
                break

            # Prefer live AirSim frame; fall back to mock
            frame = airsim_client.get_rgb_frame() if airsim_client is not None else None
            avoid_src = "AIRSIM-LIVE" if frame is not None else "synthetic-mock"
            if frame is None and mock is not None:
                frame = mock.next_frame()
            if frame is None:
                continue

            try:
                analysis = mapper.analyze(frame)
                h, w = analysis.frame_h, analysis.frame_w

                if analysis.n_blocked == 0 and analysis.n_passable == 0:
                    # No close obstacles — drift back to cruise altitude
                    _stream_z[0] = -cruise_alt
                    print(f"  [Avoid] src=[{avoid_src}] Clear — holding alt={cruise_alt:.0f}m")
                    continue

                print(f"  [Avoid] src=[{avoid_src}] Detected: {analysis.summary}")

                # A* from drone position (bottom-centre of frame) to goal (top-centre)
                waypoints_2d = plan_path(
                    analysis.obstacle_mask,
                    start=(h - 1, w // 2),
                    goal =(0,     w // 2),
                )

                if not waypoints_2d:
                    # Fully blocked — climb over the tallest object
                    max_obj_h = max(
                        (o.estimated_height_m for o in analysis.objects),
                        default=cruise_alt,
                    )
                    new_alt = max_obj_h + 3.0
                    _stream_z[0] = -new_alt
                    print(f"  [Avoid] No path — climbing to {new_alt:.1f}m")
                    continue

                # Attach per-waypoint altitude from detected object heights
                waypoints_3d = assign_altitudes(
                    waypoints_2d,
                    analysis.objects,
                    h, w,
                    default_alt_m = cruise_alt,
                    clearance_m   = 3.0,
                    min_alt_m     = cruise_alt,
                    max_alt_m     = 40.0,
                )

                # Steer toward 1/4-point of path — avoids overreacting to far waypoints
                steer_idx = max(1, len(waypoints_3d) // 4)
                _, steer_col, steer_alt = waypoints_3d[steer_idx]

                # Column deviation from frame centre → east NED nudge
                col_offset = steer_col - (w // 2)
                east_nudge = col_offset * PX_TO_M

                with ned_lock:
                    ned_target[1] += east_nudge

                _stream_z[0] = -steer_alt

                print(
                    f"  [Avoid] A* steer: col_offset={col_offset:+d}px  "
                    f"east={east_nudge:+.1f}m  alt={steer_alt:.1f}m  "
                    f"blocked={analysis.n_blocked} passable={analysis.n_passable}"
                )

            except Exception as e:
                print(f"  [Avoid] frame error: {type(e).__name__}: {e}")

    threading.Thread(target=_avoid_loop, daemon=True).start()

    # Cruise: GPS tracking loop updates ned_target[0]/[1] every second.
    # _stream reads n,e from ned_target and z from _stream_z (held at cruise alt).
    print("  [Drone] Tracking victim (GPS loop updates waypoint every 1s) ...")
    while not stop_event.is_set():
        with _lock:
            n, e = _ned_target[0], _ned_target[1]
            alt  = _state["alt"]
        print(f"  [Drone] flying  NED=({n:.1f},{e:.1f})  alt={alt:.1f}m", end="\r")
        time.sleep(1.0)

    # Land — switch PX4 to AUTO.LAND mode so it commits to a full landing
    # regardless of altitude. Setting z=0 in OFFBOARD at 50m can timeout
    # before touchdown; AUTO.LAND descends at full rate until disarm.
    # PX4 custom_mode encoding: main_mode=4 (AUTO) at bits 16-23,
    #                           sub_mode=6  (LAND) at bits 24-31
    PX4_AUTO_LAND = (4 << 16) | (6 << 24)

    print("\n  [Drone] Switching to AUTO.LAND ...")
    for _ in range(5):   # send a few times so PX4 doesn't miss it
        mav.mav.set_mode_send(
            mav.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            PX4_AUTO_LAND,
        )
        time.sleep(0.2)

    # Wait for touchdown — 60s is generous for any cruise altitude
    t0 = time.time()
    while True:
        with _lock:
            alt = _state["alt"]
        print(f"  [Drone] descending  alt={alt:5.1f}m", end="\r")
        if alt < 0.5:
            print(f"\n  [Drone] Touchdown confirmed at {alt:.1f}m")
            break
        if time.time() - t0 > 60:
            print(f"\n  [Drone] Land timeout — alt={alt:.1f}m (PX4 still descending)")
            break
        time.sleep(0.5)

    _stop_stream.set()
    print(f"  [Drone] Mission ended. Final alt={_state['alt']:.1f}m")
    return _ned_target


# ── Vision pipeline ───────────────────────────────────────────────────────────

def _preload_vision():
    """
    Load YOLOv8 classifier + CameraMock before the flight window opens.
    Returns (clf, mock) or (None, None) on failure.
    Call this in main() before starting flight so the vision thread fires
    immediately without waiting ~10s for YOLOv8 to load on GPU.
    """
    try:
        from vision.scene_classifier import get_classifier
        from vision.camera_mock import CameraMock
        print("  [Vision] Loading YOLOv8n on GPU ...")
        clf  = get_classifier(model_size="n")
        mock = CameraMock(scene="distress")
        print("  [Vision] YOLOv8n ready")
        return clf, mock
    except Exception as e:
        print(f"  [Vision] Pre-load failed: {e}")
        return None, None


def _vision_loop(
    airsim_client,
    danger_score: float,
    stop_event: threading.Event,
    clf,
    mock,
    interval_s: float = 3.0,
) -> None:
    """
    Grabs camera frame (AirSim or synthetic) and runs the full vision pipeline.
    clf and mock are pre-loaded by _preload_vision() in main() — thread starts
    instantly without waiting for model load.
    Runs in a daemon thread; prints scene intelligence output every interval_s.
    """
    if clf is None:
        print("  [Vision] Classifier not available — skipping vision loop")
        return

    frame_n = 0
    while not stop_event.is_set():
        frame_n += 1

        # Try AirSim camera first, fall back to synthetic distress frame
        frame = airsim_client.get_rgb_frame()
        if frame is not None:
            src = "AIRSIM-LIVE"
        else:
            frame = mock.next_frame()
            src   = "synthetic-mock"

        try:
            result = clf.classify(
                frame          = frame,
                danger_score   = danger_score,
                drone_dist_km  = 0.05,
                battery_pct    = 85.0,
                mission_status = "flying",
            )
            print(
                f"  [Vision] frame={frame_n:03d} src=[{src}]  "
                f"scene={result.scene_type}  threat={result.threat_level}/5  "
                f"-> {result.recommended_action[:55]}"
            )
        except Exception as e:
            print(f"  [Vision] frame={frame_n:03d} error: {type(e).__name__}: {e}")

        time.sleep(interval_s)


# ── Agent graph ────────────────────────────────────────────────────────────────

def _run_agents(
    danger_score: float,
    features: list[float],
    victim_lat: float,
    victim_lon: float,
) -> dict:
    """
    Patch agent_graph to use sim_bridge, then invoke danger_score_node through
    dispatch_node manually (avoids the full blocking loop for the demo).
    Returns state dict with station + drone assignment.
    """
    import agents.agent_graph as ag
    import simulation.sim_bridge as bridge

    # Patch drone_client calls to use sim_bridge
    ag.launch_drone  = bridge.launch_drone
    ag.get_telemetry = bridge.get_telemetry
    ag.set_waypoint  = bridge.set_waypoint
    ag.inject_fault  = bridge.inject_fault

    state: dict = {
        "danger_score": danger_score,
        "victim_lat":   victim_lat,
        "victim_lon":   victim_lon,
        "raw_features": features,
        "max_iterations": 3,    # short loop for demo
    }

    _phase_header("AGENT GRAPH — running nodes 1→5 (danger→dispatch)")

    # Node 1: danger score
    state.update(ag.danger_score_node(state))
    if not state.get("go_decision"):
        print("  [Agent 1] Score too low — no dispatch (score may need to be higher)")
        # Force go for demo if LLM fallback returned False
        state["go_decision"] = True
        state.setdefault("mission_id", "DEMO-001")

    # Node 2: SOS broadcast
    state.update(ag.sos_broadcast_node(state))

    # Node 3: station finder
    try:
        state.update(ag.station_finder_node(state))
    except Exception as e:
        print(f"  [Agent 3] DB unavailable ({type(e).__name__}) — using mock station")
        from agents.state import StationInfo
        state["nearest_stations"] = [
            StationInfo(
                station_id=1, name="Madhapur PS",
                lat=17.4410, lon=78.3830,
                distance_km=0.72, drones_available=2,
            )
        ]
        state["assigned_station"] = state["nearest_stations"][0]

    # Node 4: path planner
    state.update(ag.path_planner_node(state))

    # Node 5: dispatch
    state.update(ag.dispatch_node(state))

    return state


# ── GPS tracking loop ─────────────────────────────────────────────────────────

def _gps_track_loop(
    scene,
    ned_target: list,
    ned_lock: threading.Lock,
    stop_event: threading.Event,
    drone_ned_getter,
) -> None:
    """
    Every second: read victim GPS → compute relative NED movement → nudge drone target.
    AirSim Blocks is at Seattle; Hyderabad GPS cannot be used as absolute NED.
    Instead we track victim's movement RELATIVE to start and nudge the local target.
    """
    import math

    # Victim starting GPS (captured once so we track delta movement)
    start_lat, start_lon = scene.victim_gps()
    prev_lat,  prev_lon  = start_lat, start_lon

    while not stop_event.is_set():
        vlat, vlon = scene.victim_gps()
        alat, alon = scene.attacker_gps()

        # Delta movement since last tick (metres)
        dlat_m = (vlat - prev_lat) * 111_320.0
        dlon_m = (vlon - prev_lon) * 111_320.0 * math.cos(math.radians(vlat))
        prev_lat, prev_lon = vlat, vlon

        # Nudge local AirSim drone waypoint by same relative movement
        with ned_lock:
            ned_target[0] += dlat_m   # north
            ned_target[1] += dlon_m   # east
            dn, de = ned_target[0], ned_target[1]

        attacker_dist_m = scene.distance_m()
        phy_drone_n, phy_drone_e, _ = drone_ned_getter()
        drone_dist_m = math.sqrt((phy_drone_n - dn)**2 + (phy_drone_e - de)**2)

        print(
            f"  [GPS]  victim=({vlat:.5f}, {vlon:.5f})  "
            f"attacker={attacker_dist_m:5.1f}m away  "
            f"drone→target={drone_dist_m:6.1f}m"
        )

        if drone_dist_m < 15:
            print("  [GPS]  *** DRONE ARRIVED AT VICTIM LOCATION ***")
            stop_event.set()
            break

        time.sleep(1.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def _check_camera(airsim_client) -> bool:
    """
    Grab one frame from the AirSim front_center camera and report the result.
    Returns True if a real frame was received, False if falling back to mock.
    """
    print("\n  [Camera] Testing AirSim front_center camera ...")
    frame = airsim_client.get_rgb_frame()
    if frame is None:
        print("  [Camera] *** RESULT: None — AirSim returned no frame ***")
        print("  [Camera]     Vision will use SYNTHETIC (mock) frames")
        print("  [Camera]     Possible causes: Blocks not running, camera not")
        print("  [Camera]     configured in settings.json, or AirSim API mismatch")
        return False
    else:
        h, w, c = frame.shape
        print(f"  [Camera] *** RESULT: OK — real frame received ({w}x{h} px, {c}ch BGR) ***")
        print(f"  [Camera]     Vision will use REAL drone camera feed")
        return True


def main(fly: bool = True) -> None:
    _banner(
        "GUARDIAN DRONE — FULL INTEGRATION DEMO\n"
        "Phase 3.7 — End-to-End Pipeline Test\n"
        "Victim · Watch · Agents · Drone · Vision"
    )

    from simulation.airsim_client import AirSimClient
    from simulation.scene_setup   import AttackScene
    from simulation.biometric_sim import BiometricSimulator

    # ── 1. AirSim scene setup ─────────────────────────────────────────────────
    _phase_header("1/6  SCENE SETUP")
    airsim = AirSimClient()
    _check_camera(airsim)   # prints OK/None immediately so user knows camera status
    scene  = AttackScene(airsim)
    scene.setup()
    scene.start()   # begins continuous movement in background
    print("  [Scene] Victim and attacker actors moving continuously")

    victim_lat, victim_lon = scene.victim_gps()

    # ── 2. Pre-load vision model (before biometric wait so GPU warms up) ─────────
    _phase_header("2/6  SMARTWATCH MONITOR")
    print("  [Vision] Pre-loading YOLOv8n while waiting for attack signal ...")
    clf, mock = _preload_vision()

    sos_event = threading.Event()

    def _on_sos():
        sos_event.set()

    bio = BiometricSimulator(victim_lat, victim_lon, on_sos=_on_sos)
    bio_thread = threading.Thread(target=bio.run, daemon=True)
    bio_thread.start()

    print("  [Watch] Monitoring biometrics — baseline phase")
    print(f"  [Watch] Attack starts in 12 seconds...\n")

    # ── 3. Simulate attack event ───────────────────────────────────────────────
    time.sleep(12)
    bio.attack()   # transitions biometric phase → distress

    # ── 4. Wait for SOS confirmation ──────────────────────────────────────────
    _phase_header("3/6  WAITING FOR SOS GATE CONFIRMATION")
    print("  [Watch] Monitoring... all 5 gates must pass\n")
    sos_event.wait(timeout=60)

    if not bio.sos_detected:
        print("  [Watch] SOS timeout — forcing dispatch for demo")
        bio.sos_detected = True

    victim_lat, victim_lon = scene.victim_gps()
    danger_score = bio.latest_score
    features     = bio.latest_features
    print(f"  Victim GPS: {victim_lat:.5f}N  {victim_lon:.5f}E")
    print(f"  Danger score: {danger_score:.1f}/100")

    # ── 5. LangGraph agent pipeline ───────────────────────────────────────────
    _phase_header("4/6  AGENT PIPELINE")

    try:
        agent_state = _run_agents(danger_score, features, victim_lat, victim_lon)
        dispatch    = agent_state.get("dispatch_result")
        station     = agent_state.get("assigned_station")
        if station:
            print(f"\n  Nearest station : {station.name}  ({station.distance_km:.2f}km)")
        if dispatch:
            print(f"  Drone assigned  : {dispatch.drone_id}")
            print(f"  ETA             : {dispatch.eta_seconds:.0f}s")
    except Exception as e:
        print(f"  [Agent] Error: {e}")
        agent_state = {"dispatch_result": None}

    # ── 6. Vision pipeline (background) ───────────────────────────────────────
    _phase_header("5/6  VISION PIPELINE")
    stop_mission = threading.Event()

    # clf and mock were pre-loaded in step 2 — thread starts immediately
    vision_thread = threading.Thread(
        target=_vision_loop,
        args=(airsim, danger_score, stop_mission, clf, mock),
        daemon=True,
    )
    vision_thread.start()

    # ── 7. Drone flight ────────────────────────────────────────────────────────
    if fly:
        _phase_header("6/6  DRONE FLIGHT (PX4 OFFBOARD)")
        print("  Note: AirSim Blocks world = Seattle. Victim is 60 miles away (Yadadri).")
        print("  Drone flies 316m NE at 50m AGL in AirSim. GPS delta tracks victim.\n")

        # Local AirSim demo target — drone starts at origin, flies 50m north.
        # GPS tracker nudges this by victim's relative movement each second.
        ned_target = [300.0, 100.0, -50.0]   # 60-mile scenario: fly 316m NE at 50m alt — clears all obstacles
        ned_lock   = threading.Lock()

        # GPS tracking updates ned_target with relative victim movement
        gps_thread = threading.Thread(
            target=_gps_track_loop,
            args=(scene, ned_target, ned_lock, stop_mission,
                  AirSimClient().get_drone_ned),
            daemon=True,
        )
        gps_thread.start()

        # Flight runs in foreground — passes shared ned_target list so GPS loop
        # and drone stream thread read/write the same object.
        # airsim + mock passed so obstacle avoidance can grab live camera frames.
        _fly_to_victim(ned_target, ned_lock, stop_mission, airsim, mock)

    else:
        print("  [Drone] --no-fly flag set — skipping PX4 flight")
        print("  [Drone] Simulating 30s flight time (3x vision frames expected) ...")
        time.sleep(30)
        stop_mission.set()

    # ── 8. Mission complete ────────────────────────────────────────────────────
    _banner(
        "MISSION COMPLETE\n"
        f"  Victim GPS  : {victim_lat:.5f}N {victim_lon:.5f}E\n"
        f"  Danger score: {danger_score:.1f}/100\n"
        f"  All AI modules exercised: biometric → agents → vision → drone"
    )

    scene.stop()
    bio.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guardian Drone Integration Demo")
    parser.add_argument("--no-fly", "--no-px4", action="store_true",
                        help="Skip PX4 drone flight (agents + vision only)")
    args = parser.parse_args()
    main(fly=not args.no_fly)
