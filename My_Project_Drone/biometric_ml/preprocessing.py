"""
Phase 1.3 — Signal Preprocessing Pipeline
Phase 1.4 — Personal Baseline Calibration (PersonalBaseline class)
Phase 1.8 — Feature extraction helpers used by LSTM and XGBoost

Signals processed (Empatica E4 wrist sensor):
  BVP  64 Hz  → HRV features  (RMSSD, SDNN, pNN50, LF/HF, mean_RR, std_RR)
  EDA   4 Hz  → GSR features  (mean, std, phasic_peaks, phasic_amp, slope, tonic_mean)
  ACC  32 Hz  → Motion feat.  (mean_mag, std_mag, spectral_entropy, breathing_rate)

16 features total per 30-second window.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import linregress

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── sampling rates ───────────────────────────────────────────────────────────
FS_BVP   = 64     # Hz
FS_EDA   = 4      # Hz
FS_ACC   = 32     # Hz
FS_LABEL = 700    # Hz

# ── windowing ────────────────────────────────────────────────────────────────
WINDOW_SEC = 30   # seconds per window
STEP_SEC   = 5    # step between windows

# samples per window for each modality
WIN_BVP = WINDOW_SEC * FS_BVP    # 1920
WIN_EDA = WINDOW_SEC * FS_EDA    # 120
WIN_ACC = WINDOW_SEC * FS_ACC    # 960

# ── feature ordering (must stay fixed — XGBoost + LSTM both depend on this) ─
FEATURE_NAMES = [
    # HRV  (6)
    "hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hrv_lf_hf",
    "hrv_mean_rr", "hrv_std_rr",
    # EDA  (6)
    "eda_mean", "eda_std", "eda_phasic_peaks", "eda_phasic_amplitude",
    "eda_slope", "eda_tonic_mean",
    # ACC  (4)
    "acc_mean_mag", "acc_std_mag", "acc_spectral_entropy", "acc_breathing_rate",
]
N_FEATURES = len(FEATURE_NAMES)   # 16


# ═══════════════════════════════════════════════════════════════════════════
# 1.  FILTERS
# ═══════════════════════════════════════════════════════════════════════════

def bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    nyq  = fs / 2.0
    low  = lowcut  / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def lowpass_filter(
    signal: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter."""
    nyq = fs / 2.0
    cut = min(cutoff / nyq, 0.99)
    b, a = butter(order, cut, btype="low")
    return filtfilt(b, a, signal)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  HRV FEATURE EXTRACTION  (from BVP at 64 Hz)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_lf_hf(peaks: np.ndarray, rr: np.ndarray) -> float:
    """
    LF/HF ratio via Welch PSD on RR intervals interpolated to 4 Hz.
    LF: 0.04–0.15 Hz  |  HF: 0.15–0.40 Hz
    """
    try:
        peak_times = peaks[1:] / float(FS_BVP)          # seconds
        fs_interp  = 4.0
        t_uniform  = np.arange(peak_times[0], peak_times[-1], 1.0 / fs_interp)
        if len(t_uniform) < 8:
            return 0.0

        fn       = interp1d(peak_times, rr, kind="cubic",
                            bounds_error=False, fill_value="extrapolate")
        rr_uni   = fn(t_uniform)
        nperseg  = min(len(rr_uni), 64)
        freqs, psd = welch(rr_uni, fs=fs_interp, nperseg=nperseg)

        lf = float(np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)],
                            freqs[(freqs >= 0.04) & (freqs < 0.15)]))
        hf = float(np.trapz(psd[(freqs >= 0.15) & (freqs <= 0.40)],
                            freqs[(freqs >= 0.15) & (freqs <= 0.40)]))
        return lf / hf if hf > 1e-10 else 0.0
    except Exception:
        return 0.0


_HRV_ZERO = {k: 0.0 for k in [
    "hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hrv_lf_hf",
    "hrv_mean_rr", "hrv_std_rr",
]}


def extract_hrv_features(bvp_window: np.ndarray) -> dict:
    """
    Extract 6 HRV features from one 30-second BVP window (64 Hz).
    Returns zero dict on failure (too few peaks / flat signal).
    """
    bvp = bandpass_filter(bvp_window.flatten(), 0.5, 4.0, FS_BVP)

    # Peak detection: min 0.3 s apart (200 bpm upper bound)
    min_dist = int(0.3 * FS_BVP)
    peaks, _ = find_peaks(bvp, distance=min_dist, prominence=0.2)
    if len(peaks) < 3:
        return _HRV_ZERO.copy()

    rr = np.diff(peaks) / float(FS_BVP)          # inter-beat intervals in seconds
    rr = rr[(rr >= 0.3) & (rr <= 2.0)]           # physiologically valid (30–200 bpm)
    if len(rr) < 2:
        return _HRV_ZERO.copy()

    rr_diff = np.diff(rr)
    return {
        "hrv_rmssd":   float(np.sqrt(np.mean(rr_diff ** 2))),
        "hrv_sdnn":    float(np.std(rr, ddof=1)),
        "hrv_pnn50":   float(np.mean(np.abs(rr_diff) > 0.05)),
        "hrv_lf_hf":   _compute_lf_hf(peaks, rr),
        "hrv_mean_rr": float(np.mean(rr)),
        "hrv_std_rr":  float(np.std(rr, ddof=1)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3.  EDA / GSR FEATURE EXTRACTION  (from EDA at 4 Hz)
# ═══════════════════════════════════════════════════════════════════════════

def extract_eda_features(eda_window: np.ndarray) -> dict:
    """
    Extract 6 EDA/GSR features from one 30-second EDA window (4 Hz).

    Decomposition:
      tonic   = slow component via 0.05 Hz low-pass (skin conductance level)
      phasic  = EDA_filtered − tonic   (skin conductance response spikes)
    """
    eda = eda_window.flatten()
    eda_f = lowpass_filter(eda, 1.0, FS_EDA)          # remove high-freq noise

    # tonic/phasic decomposition — need >20 samples for second filter to be stable
    if len(eda_f) > 20:
        tonic  = lowpass_filter(eda_f, 0.05, FS_EDA)
    else:
        tonic  = np.full_like(eda_f, np.mean(eda_f))
    phasic = eda_f - tonic

    # SCR (skin conductance response) peaks in phasic component
    min_dist = max(1, int(0.5 * FS_EDA))              # at least 0.5 s apart
    peaks, props = find_peaks(phasic, distance=min_dist, height=0.005)
    peak_count = float(len(peaks))
    peak_amp   = float(np.mean(props["peak_heights"])) if len(peaks) > 0 else 0.0

    # slope via linear regression on filtered EDA
    t = np.arange(len(eda_f)) / float(FS_EDA)
    slope, *_ = linregress(t, eda_f)

    return {
        "eda_mean":             float(np.mean(eda_f)),
        "eda_std":              float(np.std(eda_f, ddof=1)),
        "eda_phasic_peaks":     peak_count,
        "eda_phasic_amplitude": peak_amp,
        "eda_slope":            float(slope),
        "eda_tonic_mean":       float(np.mean(tonic)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  ACCELEROMETER FEATURE EXTRACTION  (ACC at 32 Hz)
# ═══════════════════════════════════════════════════════════════════════════

def _spectral_entropy(signal: np.ndarray, fs: float) -> float:
    """Normalized spectral entropy (0 = single tone, 1 = white noise)."""
    try:
        _, psd    = welch(signal, fs=fs, nperseg=min(len(signal), 64))
        psd_norm  = psd / (psd.sum() + 1e-10)
        entropy   = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        max_entr  = np.log2(len(psd_norm))
        return float(entropy / max_entr) if max_entr > 0 else 0.0
    except Exception:
        return 0.0


def _breathing_rate(signal: np.ndarray) -> float:
    """Dominant frequency (0.1–0.5 Hz) → breaths per minute."""
    try:
        freqs, psd = welch(signal, fs=FS_ACC, nperseg=min(len(signal), 128))
        mask = (freqs >= 0.1) & (freqs <= 0.5)
        if not mask.any():
            return 0.0
        return float(freqs[mask][np.argmax(psd[mask])] * 60)
    except Exception:
        return 0.0


def extract_acc_features(acc_window: np.ndarray) -> dict:
    """
    Extract 4 ACC features from one 30-second window (32 Hz, shape N×3 or 3×N).
    """
    # normalise shape to (N, 3)
    if acc_window.ndim == 2 and acc_window.shape[1] == 3:
        acc = acc_window
    elif acc_window.ndim == 2 and acc_window.shape[0] == 3:
        acc = acc_window.T
    else:
        acc = acc_window.reshape(-1, 3)

    mag = np.sqrt(np.sum(acc ** 2, axis=1))           # resultant magnitude

    # breathing estimate: band-pass 0.1–0.5 Hz on magnitude
    if len(mag) >= 20:
        breath_sig = bandpass_filter(mag, 0.1, 0.5, FS_ACC)
    else:
        breath_sig = mag

    return {
        "acc_mean_mag":        float(np.mean(mag)),
        "acc_std_mag":         float(np.std(mag, ddof=1)),
        "acc_spectral_entropy": _spectral_entropy(mag, FS_ACC),
        "acc_breathing_rate":  _breathing_rate(breath_sig),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5.  COMBINED FEATURE VECTOR  (16 features)
# ═══════════════════════════════════════════════════════════════════════════

def extract_features_window(
    bvp_window: np.ndarray,
    eda_window: np.ndarray,
    acc_window: np.ndarray,
) -> np.ndarray:
    """
    Extract all 16 features from one 30-second window.
    Returns float32 array of shape (16,) ordered by FEATURE_NAMES.
    """
    feats = {
        **extract_hrv_features(bvp_window),
        **extract_eda_features(eda_window),
        **extract_acc_features(acc_window),
    }
    return np.array([feats[n] for n in FEATURE_NAMES], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  WINDOW VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def is_valid_window(
    bvp_window: np.ndarray,
    eda_window: np.ndarray,
    acc_window: np.ndarray,
    max_flat_ratio: float = 0.2,
) -> bool:
    """
    Return False if:
      - BVP or EDA has >20% consecutive-identical samples (sensor dropout)
      - any array contains NaN or Inf
    """
    for sig in [bvp_window.flatten(), eda_window.flatten()]:
        flat_ratio = float(np.mean(np.abs(np.diff(sig)) < 1e-6))
        if flat_ratio > max_flat_ratio:
            return False

    for sig in [bvp_window, eda_window, acc_window]:
        if not np.all(np.isfinite(sig)):
            return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# 7.  LABEL ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

def align_label_to_window(
    labels: np.ndarray,
    window_start_sec: float,
    window_sec: float = WINDOW_SEC,
) -> int:
    """
    Majority-vote the high-rate (700 Hz) WESAD labels over a 30-second window.
    Returns 1 (distress) if majority label == 2 (stress), else 0 (safe).
    """
    start = int(window_start_sec * FS_LABEL)
    end   = min(int((window_start_sec + window_sec) * FS_LABEL), len(labels))
    chunk = labels[start:end]
    if chunk.size == 0:
        return 0
    values, counts = np.unique(chunk, return_counts=True)
    majority = int(values[np.argmax(counts)])
    return 1 if majority == 2 else 0


# ═══════════════════════════════════════════════════════════════════════════
# 8.  SLIDING WINDOW BUILDER  (one subject → X, y)
# ═══════════════════════════════════════════════════════════════════════════

def build_windows(subject_data: dict, subject_id: int):
    """
    Slide a 30-second window (5-second step) over all wrist signals.

    Returns
    -------
    X : np.ndarray, shape (N_windows, 16), float32
    y : np.ndarray, shape (N_windows,),    int32
    """
    from biometric_ml.data_loader import get_wrist_signals, get_labels

    wrist  = get_wrist_signals(subject_data)
    labels = get_labels(subject_data)

    bvp = wrist["BVP"].flatten()
    eda = wrist["EDA"].flatten()
    acc = wrist["ACC"]                                 # (N, 3)

    # total usable duration (limited by shortest modality)
    total_sec = min(
        len(bvp) / FS_BVP,
        len(eda) / FS_EDA,
        acc.shape[0] / FS_ACC,
        len(labels) / FS_LABEL,
    )

    feature_rows, label_rows = [], []
    skipped = 0
    t = 0.0

    while t + WINDOW_SEC <= total_sec:
        bvp_w = bvp[int(t * FS_BVP): int((t + WINDOW_SEC) * FS_BVP)]
        eda_w = eda[int(t * FS_EDA): int((t + WINDOW_SEC) * FS_EDA)]
        acc_w = acc[int(t * FS_ACC): int((t + WINDOW_SEC) * FS_ACC)]

        if not is_valid_window(bvp_w, eda_w, acc_w):
            skipped += 1
            t += STEP_SEC
            continue

        feature_rows.append(extract_features_window(bvp_w, eda_w, acc_w))
        label_rows.append(align_label_to_window(labels, t))
        t += STEP_SEC

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(label_rows,   dtype=np.int32)

    n_distress = int(np.sum(y == 1))
    n_safe     = int(np.sum(y == 0))
    print(f"  S{subject_id:2d}: {len(feature_rows):4d} windows  "
          f"(skipped {skipped:3d})  distress={n_distress}  safe={n_safe}")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# 9.  PERSONAL BASELINE CALIBRATION  (Module 1.4)
# ═══════════════════════════════════════════════════════════════════════════

class PersonalBaseline:
    """
    Per-user feature normalizer.

    Fit on 2 weeks of passive-wear (baseline label) windows.
    Normalize converts features to z-scores relative to that user's own mean/std.
    Supports incremental update via exponential moving average (alpha=0.05).

    Usage
    -----
    baseline = PersonalBaseline()
    baseline.fit(X_baseline_windows)          # 2-week calibration
    X_norm = baseline.normalize(X_new)        # z-score transform
    baseline.update(X_new_window)             # incremental EMA update
    baseline.save("path/baseline.json")
    baseline = PersonalBaseline.load("path/baseline.json")
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha  = alpha      # EMA smoothing factor for incremental updates
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PersonalBaseline":
        """
        Compute per-feature mean and std from baseline windows.
        X shape: (N_windows, N_features)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.mean_ = X.mean(axis=0).astype(np.float64)
        self.std_  = X.std(axis=0, ddof=1).astype(np.float64)
        # avoid division by zero for constant features
        self.std_  = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Z-score transform X relative to this user's baseline."""
        self._check_fitted()
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def update(self, x_new: np.ndarray) -> None:
        """
        Incremental EMA update with one new window.
        Allows the baseline to slowly adapt over time (alpha=0.05 → very slow drift).
        """
        self._check_fitted()
        x = x_new.flatten().astype(np.float64)
        self.mean_ = (1 - self.alpha) * self.mean_ + self.alpha * x
        residual   = (x - self.mean_) ** 2
        self.std_  = np.sqrt(
            (1 - self.alpha) * self.std_ ** 2 + self.alpha * residual
        )
        self.std_  = np.where(self.std_ < 1e-8, 1.0, self.std_)

    def save(self, path: str | Path) -> None:
        """Persist baseline to JSON (no raw biometrics — only statistics)."""
        self._check_fitted()
        data = {
            "alpha":  self.alpha,
            "mean":   self.mean_.tolist(),
            "std":    self.std_.tolist(),
            "features": FEATURE_NAMES,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PersonalBaseline":
        """Load baseline from JSON."""
        data = json.loads(Path(path).read_text())
        obj  = cls(alpha=data["alpha"])
        obj.mean_ = np.array(data["mean"],  dtype=np.float64)
        obj.std_  = np.array(data["std"],   dtype=np.float64)
        return obj

    def _check_fitted(self) -> None:
        if self.mean_ is None:
            raise RuntimeError("PersonalBaseline not fitted yet. Call fit() first.")
