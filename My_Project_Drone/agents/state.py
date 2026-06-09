"""
Phase 2 — LangGraph 10-Agent Architecture
SharedState TypedDict + all Pydantic I/O models for agent contracts.
"""
from __future__ import annotations

import operator
import uuid
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id() -> str:
    return str(uuid.uuid4())[:8]


# ── Shared sub-objects ────────────────────────────────────────────────────────

class GPSPoint(BaseModel):
    lat: float
    lon: float
    alt_m: float = 0.0


class StationInfo(BaseModel):
    station_id: int
    name: str
    lat: float
    lon: float
    distance_km: float
    drones_available: int


class Waypoint(BaseModel):
    lat: float
    lon: float
    alt_m: float = 30.0
    speed_ms: float = 15.0


# ── Agent 1 — DangerScore output ──────────────────────────────────────────────

class DangerDecision(BaseModel):
    go: bool
    reason: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# ── Agent 2 — SOSBroadcast output ─────────────────────────────────────────────

class SOSAlert(BaseModel):
    alert_id: str = Field(default_factory=_short_id)
    priority: Literal["critical", "high", "medium"]
    message: str
    victim_location: str
    timestamp: str = Field(default_factory=_now_iso)


# ── Agent 5 — Dispatch output ────────────────────────────────────────────────

class DispatchResult(BaseModel):
    success: bool
    drone_id: str
    station_id: int
    eta_seconds: float
    error: Optional[str] = None


# ── Agent 6 — Tracking output ────────────────────────────────────────────────

class DroneTelemetry(BaseModel):
    drone_id: str
    lat: float
    lon: float
    alt_m: float
    speed_ms: float
    battery_pct: float
    status: Literal["idle", "flying", "hovering", "returning", "fault"]
    fault_code: Optional[str] = None
    camera_active: Optional[bool] = None
    motor_status: Optional[str] = None
    signal_rssi: Optional[int] = None


# ── Agent 8 — Handoff output ─────────────────────────────────────────────────

class HandoffContext(BaseModel):
    victim_lat: float
    victim_lon: float
    danger_level: float
    mission_id: str
    prev_drone_id: str
    next_drone_id: str
    summary: str


# ── Agent 9 — SceneIntelligence output ───────────────────────────────────────

class SceneClassification(BaseModel):
    scene_type: str
    threat_level: int = Field(default=3, ge=1, le=5)
    description: str
    recommended_action: str


# ── Agent 10 — ManagementNotify output ───────────────────────────────────────

class ManagementUpdate(BaseModel):
    mission_id: str
    event_type: Literal[
        "sos_triggered", "drone_dispatched", "en_route",
        "arrived", "handoff", "mission_complete", "aborted", "tick",
    ]
    status_message: str
    timestamp: str = Field(default_factory=_now_iso)


# ── SharedState — flows through all 10 LangGraph nodes ───────────────────────
# total=False: all keys are optional at any point in the pipeline
# Annotated[list, operator.add] gives append semantics for error accumulation

class SharedState(TypedDict, total=False):
    # ── Input (from ContextualGate → entry point) ─────────────────────────────
    danger_score: float           # 0–100 smoothed gate score
    victim_lat: float
    victim_lon: float
    raw_features: list[float]     # 16 biometric features (for multi-signal check)
    mission_id: str               # unique per SOS event

    # ── Agent 1 — DangerScore ─────────────────────────────────────────────────
    go_decision: bool
    danger_decision: DangerDecision

    # ── Agent 2 — SOSBroadcast ────────────────────────────────────────────────
    sos_alert: SOSAlert

    # ── Agent 3 — StationFinder ───────────────────────────────────────────────
    nearest_station: StationInfo
    backup_stations: list[StationInfo]

    # ── Agent 4 — PathPlanner ─────────────────────────────────────────────────
    flight_path: list[Waypoint]
    path_distance_km: float

    # ── Agent 5 — Dispatch ────────────────────────────────────────────────────
    dispatch_result: DispatchResult
    active_drone_id: str
    drone_server_url: str         # e.g. "http://localhost:8001"

    # ── Agent 6 — Tracking ────────────────────────────────────────────────────
    drone_telemetry: DroneTelemetry
    distance_to_victim_km: float
    mission_status: str           # idle|dispatched|flying|arrived|handoff|complete|aborted

    # ── Agent 7 — MalfunctionMonitor ──────────────────────────────────────────
    malfunction_flag: bool
    malfunction_type: str         # "" if none

    # ── Agent 8 — Handoff ─────────────────────────────────────────────────────
    handoff_context: HandoffContext
    handoff_complete: bool

    # ── Agent 9 — SceneIntelligence ───────────────────────────────────────────
    camera_frame: object          # np.ndarray from drone camera (Phase 4); None = LLM fallback
    scene_classification: SceneClassification

    # ── Agent 10 — ManagementNotify ───────────────────────────────────────────
    management_update: ManagementUpdate
    notifications_sent: int

    # ── Victim GPS update queue ───────────────────────────────────────────────
    victim_gps_queue: list[dict]  # queued GPS fixes: [{"lat": float, "lon": float}]

    # ── Latency timestamps (Unix float) ───────────────────────────────────────
    t_sos: float                  # when SOSBroadcast fired
    t_dispatched: float           # when Dispatch succeeded
    t_malfunction: float          # when MalfunctionMonitor detected fault
    t_handoff_done: float         # when Handoff completed

    # ── Loop control ──────────────────────────────────────────────────────────
    iteration: int                # monitor-loop iterations completed
    max_iterations: int           # safety cap (default 20, set lower in tests)
    errors: Annotated[list[str], operator.add]   # accumulated non-fatal errors
