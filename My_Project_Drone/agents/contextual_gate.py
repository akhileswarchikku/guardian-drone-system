"""
Phase 1.7 — Contextual Gating Layer

Sits between LSTM output and SOS dispatch. All gates must pass before a
police drone is dispatched. Purpose: raise real-world precision to 95%+
on top of the LSTM's AUC=0.90 by adding rules the ML model cannot know.

Gates (all must pass in order):
  1. Rolling smoother  — average of last 10 readings (50-second window)
  2. Score threshold   — smoothed score >= 70 (default) or 85 (in safe zone)
  3. Multi-signal      — >=2 of 3 bio-signals (HRV, GSR, breathing) in distress range
  4. Duration          — sustained above threshold for >= 15 seconds
  5. Dismiss window    — user has not double-tapped to cancel within last 10 seconds

Usage:
    gate = ContextualGate()
    gate.add_safe_zone(lat=17.385, lon=78.486, radius_m=200)  # home/work

    decision = gate.process(
        raw_score=85.0,            # LSTM distress probability * 100
        lat=17.391, lon=78.490,
        multi_signal_agree=True,
    )
    if decision.should_sos:
        dispatch_drone(...)
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import NamedTuple

import numpy as np

# ── Feature vector index map (must match biometric_ml.preprocessing.FEATURE_NAMES) ──
_IDX_HRV_RMSSD      = 0
_IDX_HRV_LF_HF      = 3
_IDX_EDA_PHASIC_PKS = 8
_IDX_EDA_TONIC_MEAN = 11
_IDX_BREATHING_RATE = 15

# ── Per-signal distress thresholds (derived from WESAD safe-class distribution) ──
_HRV_RMSSD_STRESSED  = 0.20    # rmssd below this -> HRV signal = distress
_HRV_LF_HF_STRESSED  = 2.00    # lf/hf above this -> sympathetic dominance
_EDA_TONIC_ELEVATED  = 0.90    # tonic above this -> elevated skin conductance
_EDA_PEAKS_ELEVATED  = 2       # SCR peaks above this -> arousal events
_BREATH_HIGH         = 20.0    # bpm above this -> hyperventilation
_BREATH_LOW          = 8.0     # bpm below this -> panic breath-holding


class GateDecision(NamedTuple):
    should_sos:       bool    # True only when ALL gates pass
    smoothed_score:   float   # rolling-averaged score 0–100
    blocked_by:       str     # which gate blocked, or "all_gates_passed"
    dismiss_available: bool   # True when user can still cancel via double-tap


class ContextualGate:
    """Stateful gate — call process() once per LSTM inference window (~5 seconds)."""

    SCORE_THRESHOLD      = 70.0   # default danger score needed to proceed
    SAFE_ZONE_THRESHOLD  = 85.0   # raised threshold when GPS is inside a safe zone
    DURATION_SECS        = 15.0   # must stay above threshold this long before SOS
    SMOOTHER_WINDOW      = 10     # number of readings to average (10 × 5s = 50s)
    DISMISS_SECS         = 10.0   # how long the user has to double-tap and cancel

    def __init__(self) -> None:
        self._history: deque[float] = deque(maxlen=self.SMOOTHER_WINDOW)
        self._above_since: float | None = None   # monotonic time score first crossed threshold
        self._dismissed_until: float | None = None
        self._safe_zones: list[tuple[float, float, float]] = []  # (lat, lon, radius_m)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_safe_zone(self, lat: float, lon: float, radius_m: float = 200.0) -> None:
        """Register a GPS location as a trusted safe zone (home, workplace, etc.)."""
        self._safe_zones.append((lat, lon, radius_m))

    def dismiss(self, *, now: float | None = None) -> None:
        """
        User double-tapped to cancel. Opens a 10-second window where SOS is
        suppressed even if all other gates pass. The duration timer keeps running
        so if danger persists after the window expires the system re-fires.
        """
        t = now if now is not None else time.monotonic()
        self._dismissed_until = t + self.DISMISS_SECS

    def reset(self) -> None:
        """Full state reset — call when watch comes back online after sleep."""
        self._history.clear()
        self._above_since = None
        self._dismissed_until = None

    def process(
        self,
        raw_score: float,
        lat: float | None = None,
        lon: float | None = None,
        multi_signal_agree: bool = True,
        *,
        now: float | None = None,
    ) -> GateDecision:
        """
        raw_score       : LSTM distress probability * 100  (0.0–100.0)
        lat, lon        : current GPS coordinates (None = unknown location)
        multi_signal_agree : True when >=2 of 3 bio-signals are in distress range.
                           Compute with check_multi_signal() or pass from sensor layer.
        now             : monotonic seconds override — only for unit tests.

        Returns GateDecision. Check .should_sos to decide on dispatch.
        """
        t = now if now is not None else time.monotonic()

        # Gate 1 — rolling smoother
        self._history.append(raw_score)
        smoothed = sum(self._history) / len(self._history)

        # Gate 2 — score threshold (raised inside a GPS safe zone)
        threshold = self._effective_threshold(lat, lon)
        if smoothed < threshold:
            self._above_since = None
            return GateDecision(False, smoothed, "score_below_threshold", False)

        # Gate 3 — multi-signal agreement
        if not multi_signal_agree:
            self._above_since = None
            return GateDecision(False, smoothed, "multi_signal_fail", False)

        # Gate 4 — duration: must stay above threshold continuously for DURATION_SECS
        if self._above_since is None:
            self._above_since = t
        elapsed = t - self._above_since
        if elapsed < self.DURATION_SECS:
            return GateDecision(False, smoothed, "duration_gate", False)

        # Gate 5 — dismiss window
        if self._dismissed_until is not None and t < self._dismissed_until:
            return GateDecision(False, smoothed, "dismissed", True)

        return GateDecision(True, smoothed, "all_gates_passed", True)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _effective_threshold(self, lat: float | None, lon: float | None) -> float:
        if lat is not None and lon is not None:
            for sz_lat, sz_lon, sz_radius in self._safe_zones:
                if _haversine(lat, lon, sz_lat, sz_lon) <= sz_radius:
                    return self.SAFE_ZONE_THRESHOLD
        return self.SCORE_THRESHOLD


# ── Standalone helpers ────────────────────────────────────────────────────────

def check_multi_signal(features: np.ndarray) -> bool:
    """
    Return True when >=2 of 3 bio-signal groups are in distress range.

    HRV group     : rmssd < 0.20  OR  lf_hf > 2.0
    GSR group     : eda_tonic_mean > 0.90  OR  eda_phasic_peaks > 2
    Breathing grp : acc_breathing_rate > 20  OR  < 8

    Exercise produces elevated breathing but controlled GSR and healthy HRV —
    typically scores 1/3, which fails the 2/3 requirement.
    Genuine fear typically activates all 3.
    """
    features = np.asarray(features, dtype=float)
    if features.ndim != 1 or len(features) < 16:
        raise ValueError(f"Expected 1-D array of length >=16, got shape {features.shape}")

    hrv_distress = (
        features[_IDX_HRV_RMSSD] < _HRV_RMSSD_STRESSED
        or features[_IDX_HRV_LF_HF] > _HRV_LF_HF_STRESSED
    )
    gsr_distress = (
        features[_IDX_EDA_TONIC_MEAN] > _EDA_TONIC_ELEVATED
        or features[_IDX_EDA_PHASIC_PKS] > _EDA_PEAKS_ELEVATED
    )
    breath_distress = (
        features[_IDX_BREATHING_RATE] > _BREATH_HIGH
        or features[_IDX_BREATHING_RATE] < _BREATH_LOW
    )

    return int(hrv_distress) + int(gsr_distress) + int(breath_distress) >= 2


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two GPS coordinates."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
