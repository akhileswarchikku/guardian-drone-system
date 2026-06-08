"""
Phase 1.3 — Unit tests for the signal preprocessing pipeline.

Run with:
    conda activate LLM
    cd D:/My_Project_Drone
    pytest tests/test_preprocessing.py -v

All tests use synthetic signals — no WESAD data required.
"""

import numpy as np
import pytest

from biometric_ml.preprocessing import (
    # filters
    bandpass_filter,
    lowpass_filter,
    # feature extractors
    extract_hrv_features,
    extract_eda_features,
    extract_acc_features,
    extract_features_window,
    # validation
    is_valid_window,
    # label alignment
    align_label_to_window,
    # baseline
    PersonalBaseline,
    # constants
    FEATURE_NAMES,
    N_FEATURES,
    FS_BVP, FS_EDA, FS_ACC, FS_LABEL,
    WINDOW_SEC,
)


# ── synthetic signal factories ───────────────────────────────────────────────

def make_bvp(fs: int = FS_BVP, duration: int = WINDOW_SEC, seed: int = 0) -> np.ndarray:
    """Realistic synthetic PPG: 70 bpm sine + noise."""
    rng = np.random.default_rng(seed)
    t   = np.arange(fs * duration) / fs
    sig = np.sin(2 * np.pi * (70 / 60) * t) + 0.1 * rng.standard_normal(len(t))
    return sig.astype(np.float32)


def make_eda(fs: int = FS_EDA, duration: int = WINDOW_SEC, seed: int = 1) -> np.ndarray:
    """Realistic synthetic EDA: slow tonic drift + small phasic bumps."""
    rng    = np.random.default_rng(seed)
    t      = np.arange(fs * duration) / fs
    tonic  = 2.0 + 0.01 * t
    phasic = 0.05 * np.sin(2 * np.pi * 0.1 * t)
    noise  = 0.005 * rng.standard_normal(len(t))
    return (tonic + phasic + noise).astype(np.float32)


def make_acc(fs: int = FS_ACC, duration: int = WINDOW_SEC, seed: int = 2) -> np.ndarray:
    """Realistic synthetic ACC (N, 3): resting motion + gravity on z."""
    rng = np.random.default_rng(seed)
    n   = fs * duration
    t   = np.arange(n) / fs
    x   = 0.05 * np.sin(2 * np.pi * 0.3 * t) + 0.01 * rng.standard_normal(n)
    y   = 0.05 * np.cos(2 * np.pi * 0.3 * t) + 0.01 * rng.standard_normal(n)
    z   = 1.0  + 0.03 * np.sin(2 * np.pi * 0.2 * t) + 0.01 * rng.standard_normal(n)
    return np.stack([x, y, z], axis=1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Filter tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBandpassFilter:
    def test_output_shape_preserved(self):
        sig = np.random.randn(FS_BVP * WINDOW_SEC).astype(np.float32)
        out = bandpass_filter(sig, 0.5, 4.0, FS_BVP)
        assert out.shape == sig.shape

    def test_dc_is_removed(self):
        sig = np.ones(FS_BVP * WINDOW_SEC, dtype=np.float32)
        out = bandpass_filter(sig, 0.5, 4.0, FS_BVP)
        assert abs(float(np.mean(out))) < 0.01

    def test_out_of_band_frequency_attenuated(self):
        t      = np.arange(FS_BVP * WINDOW_SEC) / FS_BVP
        # 10 Hz is above the 4 Hz cutoff
        high_f = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        out    = bandpass_filter(high_f, 0.5, 4.0, FS_BVP)
        assert np.std(out) < np.std(high_f) * 0.3


class TestLowpassFilter:
    def test_output_shape_preserved(self):
        sig = np.random.randn(FS_EDA * WINDOW_SEC).astype(np.float32)
        out = lowpass_filter(sig, 1.0, FS_EDA)
        assert out.shape == sig.shape

    def test_high_freq_attenuated(self):
        t      = np.arange(FS_EDA * WINDOW_SEC) / FS_EDA
        high_f = np.sin(2 * np.pi * 1.9 * t).astype(np.float32)  # above 1 Hz cutoff
        out    = lowpass_filter(high_f, 1.0, FS_EDA)
        assert np.std(out) < np.std(high_f) * 0.5


# ═══════════════════════════════════════════════════════════════════════════
# HRV feature tests
# ═══════════════════════════════════════════════════════════════════════════

class TestHRVFeatures:
    HRV_KEYS = {"hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hrv_lf_hf",
                "hrv_mean_rr", "hrv_std_rr"}

    def test_returns_correct_keys(self):
        feats = extract_hrv_features(make_bvp())
        assert set(feats.keys()) == self.HRV_KEYS

    def test_all_values_finite(self):
        feats = extract_hrv_features(make_bvp())
        for k, v in feats.items():
            assert np.isfinite(v), f"{k} = {v} is not finite"

    def test_rmssd_non_negative(self):
        feats = extract_hrv_features(make_bvp())
        assert feats["hrv_rmssd"] >= 0.0

    def test_sdnn_non_negative(self):
        feats = extract_hrv_features(make_bvp())
        assert feats["hrv_sdnn"] >= 0.0

    def test_pnn50_in_unit_range(self):
        feats = extract_hrv_features(make_bvp())
        assert 0.0 <= feats["hrv_pnn50"] <= 1.0

    def test_lf_hf_non_negative(self):
        feats = extract_hrv_features(make_bvp())
        assert feats["hrv_lf_hf"] >= 0.0

    def test_flat_signal_returns_all_zeros(self):
        flat  = np.zeros(FS_BVP * WINDOW_SEC, dtype=np.float32)
        feats = extract_hrv_features(flat)
        assert feats["hrv_rmssd"] == 0.0
        assert feats["hrv_sdnn"]  == 0.0

    def test_different_seeds_give_different_values(self):
        f1 = extract_hrv_features(make_bvp(seed=0))
        f2 = extract_hrv_features(make_bvp(seed=42))
        # at least one feature should differ
        assert any(abs(f1[k] - f2[k]) > 1e-8 for k in self.HRV_KEYS)


# ═══════════════════════════════════════════════════════════════════════════
# EDA feature tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEDAFeatures:
    EDA_KEYS = {"eda_mean", "eda_std", "eda_phasic_peaks", "eda_phasic_amplitude",
                "eda_slope", "eda_tonic_mean"}

    def test_returns_correct_keys(self):
        feats = extract_eda_features(make_eda())
        assert set(feats.keys()) == self.EDA_KEYS

    def test_all_values_finite(self):
        feats = extract_eda_features(make_eda())
        for k, v in feats.items():
            assert np.isfinite(v), f"{k} = {v} is not finite"

    def test_std_non_negative(self):
        feats = extract_eda_features(make_eda())
        assert feats["eda_std"] >= 0.0

    def test_peak_count_non_negative(self):
        feats = extract_eda_features(make_eda())
        assert feats["eda_phasic_peaks"] >= 0.0

    def test_mean_matches_signal_range(self):
        eda   = make_eda()
        feats = extract_eda_features(eda)
        # mean must be within the signal's observed range
        assert float(eda.min()) - 1.0 <= feats["eda_mean"] <= float(eda.max()) + 1.0

    def test_tonic_close_to_mean(self):
        eda   = make_eda()
        feats = extract_eda_features(eda)
        # tonic_mean should be near eda_mean (within 2 std)
        diff = abs(feats["eda_tonic_mean"] - feats["eda_mean"])
        assert diff < 2 * feats["eda_std"] + 0.5


# ═══════════════════════════════════════════════════════════════════════════
# ACC feature tests
# ═══════════════════════════════════════════════════════════════════════════

class TestACCFeatures:
    ACC_KEYS = {"acc_mean_mag", "acc_std_mag",
                "acc_spectral_entropy", "acc_breathing_rate"}

    def test_returns_correct_keys(self):
        feats = extract_acc_features(make_acc())
        assert set(feats.keys()) == self.ACC_KEYS

    def test_all_values_finite(self):
        feats = extract_acc_features(make_acc())
        for k, v in feats.items():
            assert np.isfinite(v), f"{k} = {v} is not finite"

    def test_mean_mag_positive(self):
        feats = extract_acc_features(make_acc())
        assert feats["acc_mean_mag"] > 0.0

    def test_spectral_entropy_in_unit_range(self):
        feats = extract_acc_features(make_acc())
        assert 0.0 <= feats["acc_spectral_entropy"] <= 1.0

    def test_breathing_rate_in_plausible_range(self):
        feats = extract_acc_features(make_acc())
        # 6–30 breaths/min is normal; we allow 0 for synthetic signals
        assert 0.0 <= feats["acc_breathing_rate"] <= 60.0

    def test_handles_transposed_input(self):
        """ACC may arrive as (3, N) — must handle gracefully."""
        acc_T = make_acc().T          # (3, N) instead of (N, 3)
        feats = extract_acc_features(acc_T)
        assert np.isfinite(feats["acc_mean_mag"])
        assert feats["acc_mean_mag"] > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Combined feature vector tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureVector:
    def test_output_shape_is_16(self):
        vec = extract_features_window(make_bvp(), make_eda(), make_acc())
        assert vec.shape == (16,)

    def test_dtype_is_float32(self):
        vec = extract_features_window(make_bvp(), make_eda(), make_acc())
        assert vec.dtype == np.float32

    def test_all_values_finite(self):
        vec = extract_features_window(make_bvp(), make_eda(), make_acc())
        assert np.all(np.isfinite(vec)), \
            f"Non-finite values at indices: {np.where(~np.isfinite(vec))[0]}"

    def test_feature_names_has_16_entries(self):
        assert len(FEATURE_NAMES) == 16
        assert N_FEATURES == 16

    def test_feature_order_is_stable(self):
        """Same input must always produce same output in same order."""
        v1 = extract_features_window(make_bvp(seed=7), make_eda(seed=7), make_acc(seed=7))
        v2 = extract_features_window(make_bvp(seed=7), make_eda(seed=7), make_acc(seed=7))
        np.testing.assert_array_equal(v1, v2)


# ═══════════════════════════════════════════════════════════════════════════
# Window validation tests
# ═══════════════════════════════════════════════════════════════════════════

class TestWindowValidation:
    def test_valid_window_passes(self):
        assert is_valid_window(make_bvp(), make_eda(), make_acc())

    def test_flat_bvp_is_rejected(self):
        flat = np.zeros(FS_BVP * WINDOW_SEC, dtype=np.float32)
        assert not is_valid_window(flat, make_eda(), make_acc())

    def test_flat_eda_is_accepted(self):
        # EDA at 4 Hz naturally has many consecutive identical values due to
        # quantisation and slow physiology — the flat-ratio check was removed for
        # EDA to avoid discarding ~90% of valid windows (e.g. WESAD subject S14).
        flat = np.zeros(FS_EDA * WINDOW_SEC, dtype=np.float32)
        assert is_valid_window(make_bvp(), flat, make_acc())

    def test_nan_in_bvp_is_rejected(self):
        bvp      = make_bvp()
        bvp[100] = np.nan
        assert not is_valid_window(bvp, make_eda(), make_acc())

    def test_inf_in_eda_is_rejected(self):
        eda    = make_eda()
        eda[5] = np.inf
        assert not is_valid_window(make_bvp(), eda, make_acc())

    def test_mostly_flat_bvp_rejected(self):
        bvp = np.zeros(FS_BVP * WINDOW_SEC, dtype=np.float32)
        # only 10% of samples are non-flat — should be rejected
        bvp[:int(0.1 * len(bvp))] = np.sin(np.arange(int(0.1 * len(bvp))) * 0.1)
        assert not is_valid_window(bvp, make_eda(), make_acc())


# ═══════════════════════════════════════════════════════════════════════════
# Label alignment tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLabelAlignment:
    def _make_labels(self, label: int, duration_sec: int = 60) -> np.ndarray:
        return np.full(FS_LABEL * duration_sec, label, dtype=np.int32)

    def test_stress_label_2_maps_to_1(self):
        assert align_label_to_window(self._make_labels(2), 0.0) == 1

    def test_baseline_label_1_maps_to_0(self):
        assert align_label_to_window(self._make_labels(1), 0.0) == 0

    def test_amusement_label_3_maps_to_0(self):
        assert align_label_to_window(self._make_labels(3), 0.0) == 0

    def test_not_defined_label_0_maps_to_0(self):
        assert align_label_to_window(self._make_labels(0), 0.0) == 0

    def test_majority_vote_baseline_wins(self):
        labels = np.zeros(FS_LABEL * 60, dtype=np.int32)
        labels[:FS_LABEL * 10] = 2     # 10s stress
        labels[FS_LABEL * 10:] = 1     # 50s baseline → baseline wins
        assert align_label_to_window(labels, 0.0) == 0

    def test_majority_vote_stress_wins(self):
        labels = np.zeros(FS_LABEL * 60, dtype=np.int32)
        labels[:FS_LABEL * 20] = 2     # 20s stress
        labels[FS_LABEL * 20:] = 1     # 40s baseline → baseline wins
        # window is only 30s — within that window: 20s stress vs 10s baseline → stress wins
        assert align_label_to_window(labels, 0.0) == 1

    def test_window_offset_correctly(self):
        labels = np.ones(FS_LABEL * 120, dtype=np.int32)    # all baseline
        labels[FS_LABEL * 60: FS_LABEL * 90] = 2            # stress at 60–90s
        assert align_label_to_window(labels, 60.0) == 1     # window at 60s → stress
        assert align_label_to_window(labels, 0.0)  == 0     # window at  0s → safe


# ═══════════════════════════════════════════════════════════════════════════
# PersonalBaseline tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPersonalBaseline:
    def _make_baseline_X(self, n: int = 200) -> np.ndarray:
        rng = np.random.default_rng(99)
        return (rng.standard_normal((n, N_FEATURES)) * 0.5 + 2.0).astype(np.float32)

    def test_fit_sets_mean_and_std(self):
        bl = PersonalBaseline()
        X  = self._make_baseline_X()
        bl.fit(X)
        assert bl.mean_ is not None
        assert bl.std_  is not None
        assert bl.mean_.shape == (N_FEATURES,)

    def test_normalize_returns_approx_zero_mean_on_training_data(self):
        bl = PersonalBaseline()
        X  = self._make_baseline_X(500)
        bl.fit(X)
        X_norm = bl.normalize(X)
        # mean should be near 0, std near 1 on training data
        np.testing.assert_allclose(X_norm.mean(axis=0), 0.0, atol=0.1)
        np.testing.assert_allclose(X_norm.std(axis=0),  1.0, atol=0.2)

    def test_normalize_stress_windows_have_large_z_scores(self):
        rng         = np.random.default_rng(5)
        X_baseline  = (rng.standard_normal((300, N_FEATURES)) * 0.5 + 2.0).astype(np.float32)
        X_stress    = (rng.standard_normal((50,  N_FEATURES)) * 0.5 + 5.0).astype(np.float32)
        bl = PersonalBaseline()
        bl.fit(X_baseline)
        z_stress = bl.normalize(X_stress)
        # mean z-score for stress windows should be >> 2
        assert float(np.abs(z_stress).mean()) > 2.0, \
            f"Expected large z-scores for shifted stress data, got {np.abs(z_stress).mean():.2f}"

    def test_update_moves_mean_incrementally(self):
        bl  = PersonalBaseline()
        X   = self._make_baseline_X()
        bl.fit(X)
        old_mean = bl.mean_.copy()
        new_x    = np.ones(N_FEATURES, dtype=np.float32) * 100.0
        bl.update(new_x)
        # mean should have shifted slightly toward 100
        assert float(np.mean(bl.mean_)) > float(np.mean(old_mean))

    def test_save_and_load_roundtrip(self, tmp_path):
        bl  = PersonalBaseline(alpha=0.03)
        bl.fit(self._make_baseline_X())
        p = tmp_path / "baseline.json"
        bl.save(p)
        bl2 = PersonalBaseline.load(p)
        np.testing.assert_allclose(bl.mean_, bl2.mean_, rtol=1e-5)
        np.testing.assert_allclose(bl.std_,  bl2.std_,  rtol=1e-5)
        assert bl2.alpha == 0.03

    def test_not_fitted_raises(self):
        bl = PersonalBaseline()
        with pytest.raises(RuntimeError, match="not fitted"):
            bl.normalize(np.zeros((1, N_FEATURES)))
