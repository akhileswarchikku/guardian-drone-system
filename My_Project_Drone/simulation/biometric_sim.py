"""
Synthetic smartwatch biometric simulator for the Guardian Drone integration demo.

Simulates an Apple Watch strapped to a victim's wrist:
  Phase 1 — baseline  : normal HR/EDA/HRV (resting)
  Phase 2 — attack    : progressive spike as attack begins
  Phase 3 — distress  : full distress signal (sustained high score)

Pipeline:
  Synthetic features (16-dim)
      → ContextualGate.process()
          → GateDecision.should_sos = True
              → SOS event fired

ContextualGate feature indices (from agents/contextual_gate.py):
  0  HRV_RMSSD      (<0.20 = distress)
  3  HRV_LF_HF      (>2.00 = distress)
  8  EDA_PHASIC_PKS (>2    = arousal)
  11 EDA_TONIC_MEAN (>0.90 = distress)
  15 BREATHING_RATE (>20   = hyperventilation)
"""
from __future__ import annotations

import random
import threading
import time
from typing import Callable

import numpy as np

from agents.contextual_gate import ContextualGate

N_FEATURES = 16      # must match biometric_ml preprocessing


class BiometricSimulator:
    """
    Generates one biometric window every WINDOW_S seconds.
    Call attack() when the simulated assault begins.
    Monitor .sos_detected to know when dispatch should trigger.

    All state is thread-safe.
    """

    WINDOW_S      = 2.0    # seconds between readings (real watch: ~5s)
    DURATION_SECS = 6.0    # ContextualGate duration gate override for demo speed

    def __init__(
        self,
        victim_lat: float,
        victim_lon: float,
        on_sos: Callable[[], None] | None = None,
    ) -> None:
        self._lat   = victim_lat
        self._lon   = victim_lon
        self._on_sos = on_sos

        self._gate = ContextualGate()
        self._gate.DURATION_SECS = self.DURATION_SECS   # faster for demo

        self._phase   = "baseline"
        self._t_phase = 0          # windows elapsed in current phase
        self._lock    = threading.Lock()

        self.sos_detected          = False
        self.latest_score: float   = 0.0
        self.latest_features: list[float] = [0.0] * N_FEATURES

        self._running = False

    # ── Public API ────────────────────────────────────────────────────────────

    def attack(self) -> None:
        """Signal that the assault has started (transitions baseline→attack)."""
        with self._lock:
            if self._phase == "baseline":
                self._phase   = "attack"
                self._t_phase = 0
                print("\n  [Watch] !!! ATTACK BEGINS — biometrics escalating !!!\n")

    def run(self) -> None:
        """
        Blocking main loop — run in a daemon thread.
        Generates windows and feeds them into the ContextualGate.
        """
        self._running = True
        while self._running:
            with self._lock:
                phase   = self._phase
                t_phase = self._t_phase
                self._t_phase += 1

            features, raw_score, multi_agree = _make_window(phase, t_phase)

            with self._lock:
                self.latest_score    = float(raw_score)
                self.latest_features = features.tolist()

            decision = self._gate.process(
                raw_score          = float(raw_score),
                lat                = self._lat,
                lon                = self._lon,
                multi_signal_agree = multi_agree,
            )

            gate_label = "ALL PASS" if decision.should_sos else decision.blocked_by
            print(
                f"  [Watch] phase={phase:9s}  "
                f"raw={raw_score:5.1f}  "
                f"smooth={decision.smoothed_score:5.1f}  "
                f"gate={gate_label}"
            )

            if decision.should_sos and not self.sos_detected:
                self.sos_detected = True
                with self._lock:
                    self._phase = "distress"
                print(
                    "\n  ╔══════════════════════════════════════════════╗"
                    "\n  ║  WATCH: SOS CONFIRMED — all 5 gates passed  ║"
                    "\n  ╚══════════════════════════════════════════════╝\n"
                )
                if self._on_sos:
                    self._on_sos()

            time.sleep(self.WINDOW_S)

    def stop(self) -> None:
        self._running = False


# ── Feature generator ─────────────────────────────────────────────────────────

def _make_window(
    phase: str,
    t: int,
) -> tuple[np.ndarray, float, bool]:
    """
    Return (features[16], danger_score 0-100, multi_signal_agree).

    Feature layout (indices that ContextualGate checks):
      0  HRV_RMSSD          3  HRV_LF_HF
      8  EDA_PHASIC_PEAKS  11  EDA_TONIC_MEAN   15  BREATHING_RATE
    """
    feat = np.zeros(N_FEATURES, dtype=np.float32)

    if phase == "baseline":
        feat[0]  = 0.48 + random.gauss(0, 0.04)   # HRV_RMSSD — healthy
        feat[3]  = 1.15 + random.gauss(0, 0.10)   # HRV_LF_HF — balanced
        feat[8]  = 0.6  + random.gauss(0, 0.15)   # EDA peaks — minimal
        feat[11] = 0.38 + random.gauss(0, 0.04)   # EDA tonic — low
        feat[15] = 13.5 + random.gauss(0, 1.0)    # Breathing — normal
        score        = random.uniform(18, 38)
        multi_agree  = False

    elif phase == "attack":
        # Progressive ramp: 0→1 over first 8 windows
        ramp = min(1.0, t / 8.0)
        feat[0]  = 0.48 - ramp * 0.32              # HRV drops sharply
        feat[3]  = 1.15 + ramp * 1.60              # LF/HF rises (sympathetic)
        feat[8]  = 0.6  + ramp * 3.8               # EDA peaks surge
        feat[11] = 0.38 + ramp * 0.72              # EDA tonic rises
        feat[15] = 13.5 + ramp * 9.0               # Hyperventilation
        score       = 28 + ramp * 65 + random.gauss(0, 4)
        multi_agree = ramp > 0.55

    else:  # distress — sustained high
        feat[0]  = 0.09 + random.gauss(0, 0.015)  # HRV very low
        feat[3]  = 3.40 + random.gauss(0, 0.20)   # LF/HF very high
        feat[8]  = 4.5  + random.gauss(0, 0.40)   # EDA peaks high
        feat[11] = 1.25 + random.gauss(0, 0.08)   # EDA tonic high
        feat[15] = 25.0 + random.gauss(0, 1.5)    # Hyperventilation
        score       = random.uniform(84, 96)
        multi_agree = True

    # Fill remaining features with plausible correlated values
    feat[1]  = max(0, feat[0] * 0.85 + random.gauss(0, 0.02))  # HRV_SDNN
    feat[2]  = max(0, feat[0] * 1.05 + random.gauss(0, 0.02))  # HRV_pNN50
    feat[4]  = max(0, feat[3] * 0.90 + random.gauss(0, 0.05))  # HF power
    feat[5]  = max(0, feat[11] * 0.55 + random.gauss(0, 0.03)) # EDA mean
    feat[6]  = max(0, feat[11] * 0.28 + random.gauss(0, 0.02)) # EDA std
    feat[7]  = max(0, feat[8]  * 0.75 + random.gauss(0, 0.10)) # EDA phasic mean
    feat[9]  = max(0, feat[11] * 0.40 + random.gauss(0, 0.03)) # EDA tonic std
    feat[10] = abs(random.gauss(0.3, 0.2))                      # ACC magnitude
    feat[12] = abs(random.gauss(0, 0.3))                        # ACC x
    feat[13] = abs(random.gauss(0, 0.3))                        # ACC y
    feat[14] = abs(random.gauss(0.1, 0.15))                     # ACC z

    return feat, float(np.clip(score, 0, 100)), multi_agree
