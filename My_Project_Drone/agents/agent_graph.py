"""
Phase 2 — LangGraph 10-Agent Architecture
Guardian Drone SOS Pipeline

Graph topology:
  START
    → [1] danger_score ──(go=False)──→ END
        │ (go=True)
    → [2] sos_broadcast
    → [3] station_finder
    → [4] path_planner
    → [5] dispatch
    → [6] tracking ←──────────────────────────────────────────┐
    → [7] malfunction_monitor                                   │
        │ (fault)                                               │
    → [8] handoff ──(handoff_ok)──────────────────────────────┘
        │ (handoff_fail)→ END
        │ (no fault)
    → [9] scene_intelligence
    → [10] management_notify
        │ (active, iteration < max)──────────────────────────┘ (back to tracking)
        │ (complete/aborted) → END

Each node returns only the state keys it owns (LangGraph merges partial updates).
LLM agents fall back to rule-based logic if OpenRouter is unavailable.
"""
from __future__ import annotations

import math
import sys
import time
import uuid
from pathlib import Path
from typing import Literal

# Allow both invocation styles:
#   python -m agents.agent_graph          (module — already works)
#   python agents/agent_graph.py          (script — needs project root on path)
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import END, StateGraph

from agents.database import (
    close_mission,
    decrement_drone,
    log_mission,
    nearest_stations,
    return_drone,
)
from agents.drone_client import get_telemetry, inject_fault, launch_drone, set_waypoint
from agents.llm_config import get_llm, with_llm_retry
from agents.state import (
    DangerDecision,
    DispatchResult,
    DroneTelemetry,
    HandoffContext,
    ManagementUpdate,
    SceneClassification,
    SharedState,
    SOSAlert,
    StationInfo,
    Waypoint,
)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _mission_id(state: SharedState) -> str:
    return state.get("mission_id") or str(uuid.uuid4())[:8]


def _server_url(state: SharedState) -> str:
    return state.get("drone_server_url", "http://localhost:8001")


# ─── Agent 1 — DangerScore ────────────────────────────────────────────────────

@with_llm_retry(max_attempts=3)
def _llm_danger_decision(score: float, features_summary: str) -> DangerDecision:
    llm = get_llm()
    structured = llm.with_structured_output(DangerDecision)
    prompt = (
        f"Guardian Drone — threat assessment.\n"
        f"Smoothed danger score: {score:.1f}/100 (threshold: 70).\n"
        f"Biometric summary: {features_summary}\n"
        f"Should we dispatch a police drone? "
        f"Respond with go=true/false, a one-sentence reason, and confidence 0-1."
    )
    return structured.invoke(prompt)


def danger_score_node(state: SharedState) -> dict:
    score   = float(state.get("danger_score", 0.0))
    mid     = _mission_id(state)
    feats   = state.get("raw_features", [])

    feat_summary = (
        f"EDA tonic={feats[8]:.2f}" if len(feats) > 8 else "no features"
    )

    try:
        decision = _llm_danger_decision(score, feat_summary)
    except Exception as e:
        decision = DangerDecision(
            go         = score >= 70.0,
            reason     = f"Rule-based fallback (LLM error: {type(e).__name__}): score={score:.1f}",
            confidence = min(1.0, max(0.0, (score - 50.0) / 50.0)),
        )

    return {
        "mission_id":     mid,
        "go_decision":    decision.go,
        "danger_decision": decision,
    }


def _route_danger_score(state: SharedState) -> str:
    return "sos_broadcast" if state.get("go_decision") else END


# ─── Agent 2 — SOSBroadcast ───────────────────────────────────────────────────

@with_llm_retry(max_attempts=3)
def _llm_sos_alert(score: float, lat: float, lon: float, reason: str) -> SOSAlert:
    llm = get_llm()
    structured = llm.with_structured_output(SOSAlert)
    prompt = (
        f"Generate a police SOS alert.\n"
        f"Danger score: {score:.0f}/100  |  Location: {lat:.5f}N, {lon:.5f}E\n"
        f"Reason: {reason}\n"
        f"Set priority to 'critical' if score>85, 'high' if >70, else 'medium'.\n"
        f"Write a clear, concise alert message (2 sentences max)."
    )
    return structured.invoke(prompt)


def sos_broadcast_node(state: SharedState) -> dict:
    score   = float(state.get("danger_score", 85.0))
    lat     = float(state.get("victim_lat", 17.44))
    lon     = float(state.get("victim_lon", 78.39))
    reason  = (state.get("danger_decision") or DangerDecision(go=True, reason="unknown")).reason

    try:
        alert = _llm_sos_alert(score, lat, lon, reason)
    except Exception as e:
        priority: Literal["critical","high","medium"] = (
            "critical" if score > 85 else "high" if score > 70 else "medium"
        )
        alert = SOSAlert(
            priority       = priority,
            message        = f"EMERGENCY: Distress detected at {lat:.4f}N {lon:.4f}E. Score={score:.0f}/100.",
            victim_location= f"{lat:.5f}N, {lon:.5f}E",
        )

    print(f"  [Agent 2] SOS ALERT [{alert.priority.upper()}] — {alert.message}")
    return {"sos_alert": alert, "t_sos": time.time()}


# ─── Agent 3 — StationFinder ─────────────────────────────────────────────────

def station_finder_node(state: SharedState) -> dict:
    lat = float(state.get("victim_lat", 17.44))
    lon = float(state.get("victim_lon", 78.39))

    stations = nearest_stations(lat, lon, limit=3)

    if not stations:
        return {
            "errors": [f"[Agent 3] No stations with available drones near ({lat:.4f},{lon:.4f})"],
        }

    nearest = StationInfo(
        station_id       = stations[0].station_id,
        name             = stations[0].name,
        lat              = stations[0].lat,
        lon              = stations[0].lon,
        distance_km      = stations[0].distance_km,
        drones_available = stations[0].drones_available,
    )
    backups = [
        StationInfo(
            station_id       = s.station_id,
            name             = s.name,
            lat              = s.lat,
            lon              = s.lon,
            distance_km      = s.distance_km,
            drones_available = s.drones_available,
        )
        for s in stations[1:]
    ]

    print(f"  [Agent 3] Nearest station: {nearest.name} ({nearest.distance_km:.2f} km)")
    return {"nearest_station": nearest, "backup_stations": backups}


# ─── Agent 4 — PathPlanner ────────────────────────────────────────────────────

def path_planner_node(state: SharedState) -> dict:
    station = state.get("nearest_station")
    vlat    = float(state.get("victim_lat", 17.44))
    vlon    = float(state.get("victim_lon", 78.39))

    if station is None:
        return {"errors": ["[Agent 4] No station found — cannot plan path"]}

    slat, slon = station.lat, station.lon

    # 3-waypoint flight plan: departure → cruise altitude → arrival
    mid_lat = (slat + vlat) / 2
    mid_lon = (slon + vlon) / 2

    path = [
        Waypoint(lat=slat,    lon=slon,    alt_m=10.0, speed_ms=10.0),  # takeoff
        Waypoint(lat=mid_lat, lon=mid_lon, alt_m=50.0, speed_ms=20.0),  # cruise
        Waypoint(lat=vlat,    lon=vlon,    alt_m=15.0, speed_ms=8.0),   # approach
    ]
    total_km = _haversine_km(slat, slon, vlat, vlon)

    print(f"  [Agent 4] Flight path: {total_km:.2f} km, {len(path)} waypoints")
    return {"flight_path": path, "path_distance_km": total_km}


# ─── Agent 5 — Dispatch ───────────────────────────────────────────────────────

def dispatch_node(state: SharedState) -> dict:
    station = state.get("nearest_station")
    path    = state.get("flight_path", [])
    mid     = _mission_id(state)
    server  = _server_url(state)

    if station is None:
        return {
            "dispatch_result": DispatchResult(
                success=False, drone_id="", station_id=0,
                eta_seconds=0, error="No station available",
            ),
            "mission_status": "aborted",
            "errors": ["[Agent 5] Dispatch aborted: no station"],
        }

    drone_id = f"GD-{station.station_id:02d}-{mid}"

    # Reserve the drone in DB
    reserved = decrement_drone(station.station_id)
    if not reserved:
        return {
            "dispatch_result": DispatchResult(
                success=False, drone_id=drone_id, station_id=station.station_id,
                eta_seconds=0, error="Station has no available drones",
            ),
            "mission_status": "aborted",
            "errors": [f"[Agent 5] Station {station.name} out of drones"],
        }

    waypoints_json = [
        {"lat": wp.lat, "lon": wp.lon, "alt_m": wp.alt_m} for wp in path
    ]

    try:
        launch_drone(
            drone_id  = drone_id,
            home_lat  = station.lat,
            home_lon  = station.lon,
            waypoints = waypoints_json,
            server_url= server,
        )
    except Exception as e:
        return_drone(station.station_id)
        return {
            "dispatch_result": DispatchResult(
                success=False, drone_id=drone_id, station_id=station.station_id,
                eta_seconds=0, error=str(e),
            ),
            "mission_status": "aborted",
            "errors": [f"[Agent 5] Drone launch failed: {e}"],
        }

    dist_km = state.get("path_distance_km", station.distance_km)
    eta     = (dist_km / 0.020)  # 20 m/s cruise → seconds

    log_mission(mid, station.station_id, drone_id)

    result = DispatchResult(
        success    = True,
        drone_id   = drone_id,
        station_id = station.station_id,
        eta_seconds= round(eta, 1),
    )

    print(f"  [Agent 5] Drone {drone_id} dispatched — ETA {eta:.0f}s")
    return {
        "dispatch_result":  result,
        "active_drone_id":  drone_id,
        "mission_status":   "flying",
        "t_dispatched":     time.time(),
    }


# ─── Agent 6 — Tracking ───────────────────────────────────────────────────────

def tracking_node(state: SharedState) -> dict:
    drone_id = state.get("active_drone_id", "")
    vlat     = float(state.get("victim_lat", 17.44))
    vlon     = float(state.get("victim_lon", 78.39))
    server   = _server_url(state)

    if not drone_id:
        return {
            "errors": ["[Agent 6] No active drone to track"],
            "mission_status": "aborted",
        }

    try:
        raw = get_telemetry(drone_id, server_url=server)
        telem = DroneTelemetry(**raw)
    except Exception as e:
        return {
            "errors": [f"[Agent 6] Telemetry error: {e}"],
            "malfunction_flag": True,
            "malfunction_type": "telemetry_lost",
        }

    dist_km = _haversine_km(telem.lat, telem.lon, vlat, vlon)

    # Arrive when within 50 m OR drone status is hovering near victim
    arrived = dist_km < 0.05 or (telem.status == "hovering" and dist_km < 0.3)

    new_status = "arrived" if arrived else state.get("mission_status", "flying")

    print(
        f"  [Agent 6] Drone {drone_id} | "
        f"pos ({telem.lat:.4f},{telem.lon:.4f}) | "
        f"dist {dist_km:.3f} km | bat {telem.battery_pct:.0f}% | "
        f"{'ARRIVED' if arrived else 'en route'}"
    )

    updates: dict = {
        "drone_telemetry":       telem,
        "distance_to_victim_km": round(dist_km, 4),
        "mission_status":        new_status,
    }

    # ── Victim GPS drift detection ─────────────────────────────────────────────
    # Pop one GPS fix from the queue each iteration.
    # If the victim moved >100m, redirect the active drone to the new position.
    gps_queue = list(state.get("victim_gps_queue") or [])
    if gps_queue:
        new_fix  = gps_queue.pop(0)
        new_vlat = float(new_fix["lat"])
        new_vlon = float(new_fix["lon"])
        drift_km = _haversine_km(vlat, vlon, new_vlat, new_vlon)
        if drift_km > 0.1:
            try:
                set_waypoint(drone_id, new_vlat, new_vlon, server_url=server)
                print(
                    f"  [Agent 6] Victim moved {drift_km*1000:.0f}m — "
                    f"waypoint updated to ({new_vlat:.5f},{new_vlon:.5f})"
                )
            except Exception as e:
                updates["errors"] = [f"[Agent 6] set_waypoint failed: {e}"]
            updates["victim_lat"] = new_vlat
            updates["victim_lon"] = new_vlon
        updates["victim_gps_queue"] = gps_queue

    return updates


# ─── Agent 7 — MalfunctionMonitor ────────────────────────────────────────────

def malfunction_monitor_node(state: SharedState) -> dict:
    telem = state.get("drone_telemetry")
    itr   = state.get("iteration", 0) + 1

    if telem is None:
        return {"iteration": itr, "malfunction_flag": False, "malfunction_type": ""}

    fault = False
    fault_type = ""

    if telem.status == "fault":
        fault, fault_type = True, telem.fault_code or "unknown_fault"
    elif telem.battery_pct <= 15.0:
        fault, fault_type = True, "battery_critical"
    elif telem.fault_code:
        fault, fault_type = True, telem.fault_code

    result: dict = {
        "iteration":        itr,
        "malfunction_flag": fault,
        "malfunction_type": fault_type,
    }

    if fault:
        print(f"  [Agent 7] MALFUNCTION detected: {fault_type} on {telem.drone_id}")
        result["t_malfunction"] = time.time()
    else:
        print(f"  [Agent 7] Drone healthy — iteration {itr}")

    return result


def _route_malfunction(state: SharedState) -> str:
    status = state.get("mission_status", "flying")
    if status == "arrived":
        return "scene_intelligence"
    if state.get("malfunction_flag"):
        return "handoff"
    return "scene_intelligence"


# ─── Agent 8 — Handoff ────────────────────────────────────────────────────────

@with_llm_retry(max_attempts=3)
def _llm_handoff_context(
    vlat: float, vlon: float, score: float, mid: str,
    prev_id: str, next_id: str,
) -> HandoffContext:
    llm = get_llm()
    structured = llm.with_structured_output(HandoffContext)
    prompt = (
        f"Drone handoff required.\n"
        f"Mission {mid}: drone {prev_id} malfunctioned.\n"
        f"Replacement drone: {next_id}.\n"
        f"Victim at ({vlat:.5f}, {vlon:.5f}), danger level {score:.0f}/100.\n"
        f"Write a brief handoff summary for the new drone operator."
    )
    return structured.invoke(prompt)


def handoff_node(state: SharedState) -> dict:
    backups  = state.get("backup_stations", [])
    mid      = _mission_id(state)
    prev_id  = state.get("active_drone_id", "unknown")
    score    = float(state.get("danger_score", 85.0))
    vlat     = float(state.get("victim_lat", 17.44))
    vlon     = float(state.get("victim_lon", 78.39))
    server   = _server_url(state)
    path     = state.get("flight_path", [])

    if not backups:
        print(f"  [Agent 8] No backup stations — mission cannot continue")
        return {
            "handoff_complete": False,
            "mission_status":   "aborted",
            "errors":           [f"[Agent 8] No backup stations for handoff on mission {mid}"],
        }

    backup    = backups[0]
    new_id    = f"GD-{backup.station_id:02d}-{mid}-B"
    reserved  = decrement_drone(backup.station_id)

    if not reserved:
        return {
            "handoff_complete": False,
            "mission_status":   "aborted",
            "errors":           [f"[Agent 8] Backup station {backup.name} has no drones"],
        }

    waypoints_json = [
        {"lat": wp.lat, "lon": wp.lon, "alt_m": wp.alt_m} for wp in path
    ]

    try:
        launch_drone(
            drone_id  = new_id,
            home_lat  = backup.lat,
            home_lon  = backup.lon,
            waypoints = waypoints_json,
            server_url= server,
        )
    except Exception as e:
        return_drone(backup.station_id)
        return {
            "handoff_complete": False,
            "mission_status":   "aborted",
            "errors":           [f"[Agent 8] Backup launch failed: {e}"],
        }

    try:
        ctx = _llm_handoff_context(vlat, vlon, score, mid, prev_id, new_id)
    except Exception:
        ctx = HandoffContext(
            victim_lat   = vlat,
            victim_lon   = vlon,
            danger_level = score,
            mission_id   = mid,
            prev_drone_id= prev_id,
            next_drone_id= new_id,
            summary      = f"Handoff from {prev_id} to {new_id}. Victim at ({vlat:.4f},{vlon:.4f}).",
        )

    print(f"  [Agent 8] Handoff: {prev_id} -> {new_id} ({backup.name})")
    return {
        "handoff_context":  ctx,
        "handoff_complete": True,
        "active_drone_id":  new_id,
        "malfunction_flag": False,
        "malfunction_type": "",
        "mission_status":   "flying",
        "t_handoff_done":   time.time(),
    }


def _route_handoff(state: SharedState) -> str:
    return "tracking" if state.get("handoff_complete") else END


# ─── Agent 9 — SceneIntelligence ─────────────────────────────────────────────

@with_llm_retry(max_attempts=3)
def _llm_scene_classification(
    dist_km: float, battery: float, mission_status: str, score: float,
) -> SceneClassification:
    llm = get_llm()
    structured = llm.with_structured_output(SceneClassification)
    prompt = (
        f"Guardian Drone — scene classification.\n"
        f"Drone distance to victim: {dist_km:.3f} km | "
        f"Battery: {battery:.0f}% | "
        f"Mission status: {mission_status} | "
        f"Danger score: {score:.0f}/100\n"
        f"Classify the scene type (e.g. 'outdoor_assault', 'isolated_area', 'crowded_public'), "
        f"threat_level 1-5, and recommended action."
    )
    return structured.invoke(prompt)


def scene_intelligence_node(state: SharedState) -> dict:
    telem   = state.get("drone_telemetry")
    dist    = float(state.get("distance_to_victim_km", 999.0))
    score   = float(state.get("danger_score", 85.0))
    status  = state.get("mission_status", "flying")
    battery = telem.battery_pct if telem else 100.0

    # ── Vision path (Phase 4): use camera frame if available ──────────────────
    camera_frame = state.get("camera_frame")
    if camera_frame is not None:
        try:
            from vision.scene_classifier import get_classifier
            clf = get_classifier(model_size="n")   # nano for speed in pipeline
            classification = clf.classify(
                frame          = camera_frame,
                danger_score   = score,
                drone_dist_km  = dist,
                battery_pct    = battery,
                mission_status = status,
            )
            print(
                f"  [Agent 9] [VISION] Scene: {classification.scene_type} | "
                f"threat {classification.threat_level}/5"
            )
            return {"scene_classification": classification}
        except Exception as e:
            print(f"  [Agent 9] Vision failed ({type(e).__name__}), falling back to LLM")

    # ── LLM fallback (Phase 2 behaviour, no camera feed) ──────────────────────
    try:
        classification = _llm_scene_classification(dist, battery, status, score)
    except Exception:
        threat = 5 if score >= 85 else 4 if score >= 75 else 3
        classification = SceneClassification(
            scene_type        = "unknown_outdoor",
            threat_level      = threat,
            description       = f"Drone {dist:.2f}km from victim. Battery {battery:.0f}%.",
            recommended_action= "Maintain visual contact and await ground units.",
        )

    print(f"  [Agent 9] [LLM] Scene: {classification.scene_type} | threat level {classification.threat_level}/5")
    return {"scene_classification": classification}


# ─── Agent 10 — ManagementNotify ─────────────────────────────────────────────

@with_llm_retry(max_attempts=3)
def _llm_management_update(
    mid: str, status: str, score: float, itr: int, dist_km: float,
) -> ManagementUpdate:
    llm = get_llm()
    structured = llm.with_structured_output(ManagementUpdate)

    event_map = {
        "arrived":  "arrived",
        "complete": "mission_complete",
        "aborted":  "aborted",
        "handoff":  "handoff",
    }
    event = event_map.get(status, "tick")

    prompt = (
        f"Generate a management status update.\n"
        f"Mission {mid} | Status: {status} | Iteration: {itr} | "
        f"Distance to victim: {dist_km:.3f} km | Danger score: {score:.0f}\n"
        f"Write a concise (1-sentence) status update for supervisors."
    )
    result = structured.invoke(prompt)
    result.mission_id = mid
    return result


def management_notify_node(state: SharedState) -> dict:
    mid    = _mission_id(state)
    status = state.get("mission_status", "flying")
    score  = float(state.get("danger_score", 85.0))
    itr    = state.get("iteration", 1)
    dist   = float(state.get("distance_to_victim_km", 999.0))
    sent   = state.get("notifications_sent", 0) + 1
    mid_   = _mission_id(state)

    if status == "arrived":
        new_status = "complete"
        close_mission(mid_)
    else:
        new_status = status

    try:
        update = _llm_management_update(mid, status, score, itr, dist)
    except Exception:
        event_map = {
            "arrived":  "arrived",
            "complete": "mission_complete",
            "aborted":  "aborted",
        }
        event = event_map.get(status, "tick")
        update = ManagementUpdate(
            mission_id    = mid,
            event_type    = event,
            status_message= f"Mission {mid}: {status} | dist={dist:.2f}km | itr={itr}",
        )

    print(f"  [Agent 10] [{update.event_type.upper()}] {update.status_message}")
    return {
        "management_update":  update,
        "notifications_sent": sent,
        "mission_status":     new_status,
    }


def _route_management_notify(state: SharedState) -> str:
    status   = state.get("mission_status", "flying")
    itr      = state.get("iteration", 0)
    max_itr  = state.get("max_iterations", 20)

    if status in ("complete", "aborted"):
        return END
    if itr >= max_itr:
        print(f"  [management_notify] max_iterations={max_itr} reached — terminating loop")
        return END
    return "tracking"


# ─── Build graph ──────────────────────────────────────────────────────────────

def build_graph():
    wf = StateGraph(SharedState)

    wf.add_node("danger_score",        danger_score_node)
    wf.add_node("sos_broadcast",       sos_broadcast_node)
    wf.add_node("station_finder",      station_finder_node)
    wf.add_node("path_planner",        path_planner_node)
    wf.add_node("dispatch",            dispatch_node)
    wf.add_node("tracking",            tracking_node)
    wf.add_node("malfunction_monitor", malfunction_monitor_node)
    wf.add_node("handoff",             handoff_node)
    wf.add_node("scene_intelligence",  scene_intelligence_node)
    wf.add_node("management_notify",   management_notify_node)

    wf.set_entry_point("danger_score")

    wf.add_conditional_edges("danger_score", _route_danger_score)

    wf.add_edge("sos_broadcast",  "station_finder")
    wf.add_edge("station_finder", "path_planner")
    wf.add_edge("path_planner",   "dispatch")
    wf.add_edge("dispatch",       "tracking")
    wf.add_edge("tracking",       "malfunction_monitor")

    wf.add_conditional_edges("malfunction_monitor", _route_malfunction)

    wf.add_edge("scene_intelligence", "management_notify")
    wf.add_conditional_edges("management_notify",   _route_management_notify)
    wf.add_conditional_edges("handoff",             _route_handoff)

    return wf.compile()


# Singleton graph (lazy-initialized per import)
graph = build_graph()


# ─── CLI helper ───────────────────────────────────────────────────────────────

def run_sos(
    danger_score: float,
    victim_lat:   float,
    victim_lon:   float,
    max_iterations: int = 20,
    drone_server_url: str = "http://localhost:8001",
    raw_features: list[float] | None = None,
) -> SharedState:
    """Invoke the full 10-agent pipeline and return final state."""
    initial: SharedState = {
        "danger_score":     danger_score,
        "victim_lat":       victim_lat,
        "victim_lon":       victim_lon,
        "raw_features":     raw_features or [],
        "max_iterations":   max_iterations,
        "drone_server_url": drone_server_url,
        "errors":           [],
    }
    return graph.invoke(initial)


if __name__ == "__main__":
    score = float(sys.argv[1]) if len(sys.argv) > 1 else 85.0
    print(f"\nRunning SOS pipeline with danger_score={score}")
    print("(Ensure agents/drone_mock.py is running on port 8001)\n")
    final = run_sos(
        danger_score=score,
        victim_lat=17.4490,
        victim_lon=78.3878,
        max_iterations=5,
    )
    print(f"\nFinal mission_status: {final.get('mission_status')}")
    print(f"Notifications sent: {final.get('notifications_sent', 0)}")
    if final.get("errors"):
        print("Errors:", final["errors"])
