"""
Tests for Phase 1.7 — Contextual Gating Layer.

All tests use injectable `now` timestamps so no real sleeping is needed.
The two mandatory spec scenarios are marked with "SPEC TEST".
"""

import numpy as np
import pytest

from agents.contextual_gate import ContextualGate, GateDecision, check_multi_signal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _exercise_features() -> np.ndarray:
    """
    Simulates a jogging session: elevated breathing, healthy HRV, low-normal GSR.
    Expected multi-signal result: 1/3 (breathing only) -> agree=False.
    """
    f = np.zeros(16)
    f[0]  = 0.38   # hrv_rmssd    — above 0.20 threshold (healthy HRV during exercise)
    f[3]  = 1.1    # hrv_lf_hf   — below 2.0 (no sympathetic dominance)
    f[8]  = 1      # eda_phasic_peaks — <= 2 (not elevated)
    f[11] = 0.55   # eda_tonic_mean  — below 0.90 (low GSR, not fear-sweat)
    f[15] = 26.0   # acc_breathing_rate — above 20 (exercise breathing)
    return f


def _distress_features() -> np.ndarray:
    """
    Simulates genuine fear: low HRV, elevated GSR, hyperventilation.
    Expected multi-signal result: 3/3 -> agree=True.
    """
    f = np.zeros(16)
    f[0]  = 0.12   # hrv_rmssd    — below 0.20 (stress-suppressed HRV)
    f[3]  = 3.2    # hrv_lf_hf   — above 2.0 (sympathetic dominance)
    f[8]  = 4      # eda_phasic_peaks — above 2 (SCR events)
    f[11] = 1.5    # eda_tonic_mean  — above 0.90 (elevated skin conductance)
    f[15] = 22.0   # acc_breathing_rate — above 20 (hyperventilation)
    return f


def _feed_n(gate, score, agree, n_readings, t_start, t_step=5.0):
    """Feed n readings with given score/agree, advancing clock by t_step each time."""
    last = None
    for i in range(n_readings):
        t = t_start + i * t_step
        last = gate.process(score, multi_signal_agree=agree, now=t)
    return last


# ── check_multi_signal unit tests ─────────────────────────────────────────────

class TestCheckMultiSignal:

    def test_exercise_features_not_agreement(self):
        # SPEC TEST: exercise = elevated breathing only -> 1/3 -> False
        assert check_multi_signal(_exercise_features()) is False

    def test_distress_features_agreement(self):
        # SPEC TEST: all 3 signals in distress range -> True
        assert check_multi_signal(_distress_features()) is True

    def test_hrv_alone_not_enough(self):
        f = np.zeros(16)
        f[0]  = 0.10   # rmssd low -> HRV distress
        f[11] = 0.50   # tonic normal
        f[15] = 15.0   # breathing normal
        assert check_multi_signal(f) is False  # 1/3

    def test_gsr_and_breathing_is_enough(self):
        f = np.zeros(16)
        f[0]  = 0.40   # rmssd fine
        f[3]  = 0.80   # lf_hf fine
        f[11] = 1.20   # tonic elevated -> GSR distress
        f[15] = 25.0   # breathing elevated -> Breathing distress
        assert check_multi_signal(f) is True   # 2/3

    def test_all_safe_is_false(self):
        f = np.zeros(16)
        f[0]  = 0.35; f[3] = 0.9; f[11] = 0.60; f[15] = 14.0
        assert check_multi_signal(f) is False

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            check_multi_signal(np.zeros(10))


# ── ContextualGate unit tests ─────────────────────────────────────────────────

class TestScoreThreshold:

    def test_score_below_threshold_blocked(self):
        gate = ContextualGate()
        d = gate.process(65.0, multi_signal_agree=True, now=0.0)
        assert not d.should_sos
        assert d.blocked_by == "score_below_threshold"

    def test_score_at_threshold_passes(self):
        # Still needs duration gate — just verify score gate passes
        gate = ContextualGate()
        d = gate.process(70.0, multi_signal_agree=True, now=0.0)
        assert d.blocked_by == "duration_gate"   # score passed, duration blocking

    def test_score_above_threshold_passes(self):
        gate = ContextualGate()
        d = gate.process(85.0, multi_signal_agree=True, now=0.0)
        assert d.blocked_by == "duration_gate"


class TestMultiSignalGate:

    def test_no_agreement_blocked(self):
        gate = ContextualGate()
        d = gate.process(80.0, multi_signal_agree=False, now=0.0)
        assert not d.should_sos
        assert d.blocked_by == "multi_signal_fail"

    def test_no_agreement_resets_duration_timer(self):
        gate = ContextualGate()
        gate.process(80.0, multi_signal_agree=True, now=0.0)   # start timer
        gate.process(80.0, multi_signal_agree=False, now=8.0)  # multi-signal fails → reset
        gate.process(80.0, multi_signal_agree=True, now=9.0)   # restart timer from here
        d = gate.process(80.0, multi_signal_agree=True, now=20.0)  # only 11s since restart
        assert d.blocked_by == "duration_gate"


class TestDurationGate:

    def test_too_short_blocked(self):
        gate = ContextualGate()
        gate.process(80.0, multi_signal_agree=True, now=0.0)
        d = gate.process(80.0, multi_signal_agree=True, now=10.0)
        assert d.blocked_by == "duration_gate"

    def test_exactly_15s_fires(self):
        gate = ContextualGate()
        gate.process(80.0, multi_signal_agree=True, now=0.0)
        d = gate.process(80.0, multi_signal_agree=True, now=15.0)
        assert d.should_sos
        assert d.blocked_by == "all_gates_passed"

    def test_score_drop_resets_timer(self):
        gate = ContextualGate()
        # t=0: history=[80], smooth=80 -> timer starts
        gate.process(80.0, multi_signal_agree=True, now=0.0)
        # t=10: history=[80,50], smooth=65 < 70 -> timer resets
        gate.process(50.0, multi_signal_agree=True, now=10.0)
        assert gate._above_since is None
        # t=11: history=[80,50,80], smooth=70.0 -> timer restarts here
        gate.process(80.0, multi_signal_agree=True, now=11.0)
        # t=22: elapsed=11s since restart (< 15) -> still blocked
        d = gate.process(80.0, multi_signal_agree=True, now=22.0)
        assert d.blocked_by == "duration_gate"
        # t=26.5: elapsed=15.5s since t=11 -> fires
        d = gate.process(80.0, multi_signal_agree=True, now=26.5)
        assert d.should_sos


class TestRollingSmoother:

    def test_single_spike_smoothed_below_threshold(self):
        gate = ContextualGate()
        # 9 safe readings, then 1 spike
        for i in range(9):
            gate.process(0.0, multi_signal_agree=True, now=float(i * 5))
        d = gate.process(100.0, multi_signal_agree=True, now=45.0)
        # smoothed = (0*9 + 100) / 10 = 10.0 — well below 70
        assert not d.should_sos
        assert d.smoothed_score == pytest.approx(10.0)

    def test_sustained_high_scores_accumulate(self):
        gate = ContextualGate()
        # Feed 10 readings of 80 — smoother should saturate at 80
        for i in range(10):
            gate.process(80.0, multi_signal_agree=True, now=float(i * 5))
        assert gate._history.count(80.0) == 10

    def test_smoothed_score_is_average(self):
        gate = ContextualGate()
        gate.process(60.0, multi_signal_agree=True, now=0.0)
        d = gate.process(80.0, multi_signal_agree=True, now=5.0)
        assert d.smoothed_score == pytest.approx(70.0)


class TestGPSSafeZone:

    HOME_LAT, HOME_LON = 17.385, 78.486

    def _gate_with_home(self):
        gate = ContextualGate()
        gate.add_safe_zone(self.HOME_LAT, self.HOME_LON, radius_m=200.0)
        return gate

    def test_inside_safe_zone_raises_threshold(self):
        gate = self._gate_with_home()
        # score=75 passes default threshold (70) but NOT safe-zone threshold (85)
        gate.process(75.0, lat=self.HOME_LAT, lon=self.HOME_LON,
                     multi_signal_agree=True, now=0.0)
        d = gate.process(75.0, lat=self.HOME_LAT, lon=self.HOME_LON,
                         multi_signal_agree=True, now=15.0)
        assert not d.should_sos
        assert d.blocked_by == "score_below_threshold"

    def test_outside_safe_zone_uses_normal_threshold(self):
        gate = self._gate_with_home()
        # 5km away — not in safe zone
        gate.process(75.0, lat=17.430, lon=78.480,
                     multi_signal_agree=True, now=0.0)
        d = gate.process(75.0, lat=17.430, lon=78.480,
                         multi_signal_agree=True, now=15.0)
        assert d.should_sos

    def test_unknown_gps_uses_normal_threshold(self):
        gate = self._gate_with_home()
        gate.process(75.0, lat=None, lon=None,
                     multi_signal_agree=True, now=0.0)
        d = gate.process(75.0, lat=None, lon=None,
                         multi_signal_agree=True, now=15.0)
        assert d.should_sos

    def test_border_of_safe_zone(self):
        gate = self._gate_with_home()
        # ~199m north of home — inside 200m radius
        lat_199m_north = self.HOME_LAT + (199.0 / 111_320.0)
        d = gate.process(75.0, lat=lat_199m_north, lon=self.HOME_LON,
                         multi_signal_agree=True, now=0.0)
        assert d.blocked_by == "score_below_threshold"  # safe-zone threshold applies


class TestDismissWindow:

    def test_dismiss_blocks_trigger(self):
        gate = ContextualGate()
        # Build up to trigger
        gate.process(80.0, multi_signal_agree=True, now=0.0)
        d_trigger = gate.process(80.0, multi_signal_agree=True, now=15.0)
        assert d_trigger.should_sos

        # User dismisses immediately
        gate.dismiss(now=15.0)
        d_after = gate.process(80.0, multi_signal_agree=True, now=16.0)
        assert not d_after.should_sos
        assert d_after.blocked_by == "dismissed"
        assert d_after.dismiss_available is True

    def test_dismiss_expires_and_refires(self):
        gate = ContextualGate()
        # Build up and fire SOS at t=15
        gate.process(80.0, multi_signal_agree=True, now=0.0)
        gate.process(80.0, multi_signal_agree=True, now=15.0)
        gate.dismiss(now=15.0)  # dismiss; window expires at t=25, timer keeps running

        # Within dismiss window: blocked
        d_blocked = gate.process(80.0, multi_signal_agree=True, now=20.0)
        assert not d_blocked.should_sos
        assert d_blocked.blocked_by == "dismissed"

        # After dismiss expires and danger persists: re-fires
        d_refire = gate.process(80.0, multi_signal_agree=True, now=26.0)
        assert d_refire.should_sos


# ── SPEC TESTS (mandatory per checklist) ─────────────────────────────────────

class TestSpecScenarios:

    def test_exercise_scenario_must_not_trigger(self):
        """
        SPEC TEST 1 — Exercise scenario.
        Simulates: high score from elevated breathing + HR, but GSR is low.
        Multi-signal check returns False (only 1/3 signals in distress range).
        Gate must block for full 20s sustained period.
        """
        assert check_multi_signal(_exercise_features()) is False, \
            "exercise features should not satisfy multi-signal agreement"

        gate = ContextualGate()
        agree = check_multi_signal(_exercise_features())
        # LSTM gives moderate score during exercise (low EDA keeps it below 70)
        results = []
        for i in range(5):   # 20 seconds: t=0,5,10,15,20
            d = gate.process(60.0, multi_signal_agree=agree, now=float(i * 5))
            results.append(d)

        assert not any(d.should_sos for d in results), \
            "exercise scenario must NOT trigger SOS"

    def test_genuine_distress_must_trigger(self):
        """
        SPEC TEST 2 — Genuine distress scenario.
        Simulates: all 3 bio-signals in distress range, unknown GPS, score=85, 20s sustained.
        Gate must fire SOS.
        """
        assert check_multi_signal(_distress_features()) is True, \
            "distress features should satisfy multi-signal agreement"

        gate = ContextualGate()
        agree = check_multi_signal(_distress_features())
        triggered = False
        for i in range(5):   # 20 seconds: t=0,5,10,15,20
            d = gate.process(85.0, lat=None, lon=None,
                             multi_signal_agree=agree, now=float(i * 5))
            if d.should_sos:
                triggered = True
                break

        assert triggered, "genuine distress scenario MUST trigger SOS"

    def test_false_positive_rate_gate(self):
        """
        SPEC: score=60, single signal only -> gate blocks -> no dispatch.
        Matches checklist Test 2.
        """
        gate = ContextualGate()
        d = gate.process(60.0, multi_signal_agree=False, now=0.0)
        assert not d.should_sos
        # Keep feeding for 30s — should never trigger
        for i in range(1, 7):
            d = gate.process(60.0, multi_signal_agree=False, now=float(i * 5))
        assert not d.should_sos
