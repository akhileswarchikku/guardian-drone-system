"""
Apple Watch -> Guardian Drone Baseline Predictor

Maps Apple Health export data to the 16 WESAD features and runs predictions
through the saved XGBoost baseline model.

Two modes:
  ECG mode  — 3 x 30s ECG files -> true R-peak HRV -> high quality
  HR  mode  — continuous HR records -> estimated RR -> many windows over time

Feature coverage:
  HRV (6)  : computed from ECG R-peaks or estimated from HR readings
  EDA (6)  : imputed with WESAD safe-class medians (no GSR sensor on Apple Watch)
  ACC (4)  : acc_mean_mag/std from PhysicalEffort, breathing_rate from RespiratoryRate,
             acc_spectral_entropy imputed

Run:
    conda activate LLM_GPU
    cd D:/My_Project_Drone
    python -m biometric_ml.apple_watch_adapter
"""

from __future__ import annotations
import pickle
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch
from scipy.interpolate import interp1d

ROOT        = Path(__file__).parent.parent
AW_DIR      = ROOT / "data" / "raw" / "apple_watch"
MODELS_DIR  = ROOT / "biometric_ml" / "models"
DOCS_DIR    = ROOT / "docs"

FEATURE_NAMES = [
    "hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hrv_lf_hf",
    "hrv_mean_rr", "hrv_std_rr",
    "eda_mean", "eda_std", "eda_phasic_peaks", "eda_phasic_amplitude",
    "eda_slope", "eda_tonic_mean",
    "acc_mean_mag", "acc_std_mag", "acc_spectral_entropy", "acc_breathing_rate",
]

# WESAD safe-class medians — used to impute unavailable sensors
SAFE_MEDIANS = {
    "hrv_rmssd"          : 0.302602,
    "hrv_sdnn"           : 0.230648,
    "hrv_pnn50"          : 0.705882,
    "hrv_lf_hf"          : 0.458764,
    "hrv_mean_rr"        : 0.826339,
    "hrv_std_rr"         : 0.230648,
    "eda_mean"           : 0.634472,
    "eda_std"            : 0.011560,
    "eda_phasic_peaks"   : 3.000000,
    "eda_phasic_amplitude": 0.009678,
    "eda_slope"          : -0.000223,
    "eda_tonic_mean"     : 0.634157,
    "acc_mean_mag"       : 63.360291,
    "acc_std_mag"        : 1.356358,
    "acc_spectral_entropy": 0.915739,
    "acc_breathing_rate" : 15.000000,
}

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ─────────────────────────────────────────────────────────────────────────────
# HRV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hrv_from_rr(rr: np.ndarray) -> dict:
    """Compute 6 HRV features from a clean RR interval array (seconds)."""
    if len(rr) < 3:
        return {k: SAFE_MEDIANS[k] for k in ["hrv_rmssd","hrv_sdnn","hrv_pnn50",
                                               "hrv_lf_hf","hrv_mean_rr","hrv_std_rr"]}
    rr_diff = np.diff(rr)
    rmssd   = float(np.sqrt(np.mean(rr_diff ** 2)))
    sdnn    = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0
    pnn50   = float(np.mean(np.abs(rr_diff) > 0.05))
    mean_rr = float(np.mean(rr))
    std_rr  = sdnn

    # LF/HF via Welch on uniformly resampled RR tachogram
    lf_hf = 0.0
    if len(rr) >= 6:
        try:
            beat_times = np.cumsum(np.concatenate([[0], rr[:-1]]))
            fs = 4.0
            t_uni = np.arange(beat_times[0], beat_times[-1], 1.0 / fs)
            if len(t_uni) >= 8:
                fn = interp1d(beat_times, rr, kind="linear",
                              bounds_error=False, fill_value="extrapolate")
                rr_uni = fn(t_uni)
                nperseg = min(len(rr_uni), 64)
                freqs, psd = welch(rr_uni, fs=fs, nperseg=nperseg)
                lf = float(_trapz(psd[(freqs >= 0.04) & (freqs < 0.15)],
                                   freqs[(freqs >= 0.04) & (freqs < 0.15)]))
                hf = float(_trapz(psd[(freqs >= 0.15) & (freqs <= 0.40)],
                                   freqs[(freqs >= 0.15) & (freqs <= 0.40)]))
                lf_hf = min(lf / hf, 20.0) if hf > 1e-10 else 0.0
        except Exception:
            pass

    return {
        "hrv_rmssd"  : rmssd,
        "hrv_sdnn"   : sdnn,
        "hrv_pnn50"  : pnn50,
        "hrv_lf_hf"  : lf_hf,
        "hrv_mean_rr": mean_rr,
        "hrv_std_rr" : std_rr,
    }


def _hrv_from_ecg(ecg: np.ndarray, fs: float = 512.0) -> dict:
    """Detect R-peaks in a Lead-I ECG and compute HRV."""
    # normalise
    ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-9)
    min_dist = int(0.3 * fs)          # 300 ms refractory period
    peaks, _ = find_peaks(ecg, distance=min_dist, height=0.4)
    if len(peaks) < 3:
        # try inverted
        peaks, _ = find_peaks(-ecg, distance=min_dist, height=0.4)
    if len(peaks) < 3:
        return {k: SAFE_MEDIANS[k] for k in ["hrv_rmssd","hrv_sdnn","hrv_pnn50",
                                               "hrv_lf_hf","hrv_mean_rr","hrv_std_rr"]}
    rr_all = np.diff(peaks) / fs
    valid  = (rr_all >= 0.3) & (rr_all <= 2.0)
    rr     = rr_all[valid]
    return _hrv_from_rr(rr)


def _hrv_from_hr_window(hr_bpm: np.ndarray) -> dict:
    """Estimate HRV features from a window of HR samples (bpm)."""
    hr_bpm = hr_bpm[(hr_bpm > 30) & (hr_bpm < 220)]
    if len(hr_bpm) < 2:
        return {k: SAFE_MEDIANS[k] for k in ["hrv_rmssd","hrv_sdnn","hrv_pnn50",
                                               "hrv_lf_hf","hrv_mean_rr","hrv_std_rr"]}
    rr = 60.0 / hr_bpm
    return _hrv_from_rr(rr)


# ─────────────────────────────────────────────────────────────────────────────
# Apple Health XML parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_health_xml(xml_path: Path) -> dict:
    """
    Parse export.xml and return time-indexed arrays for each relevant type.
    Returns dict with keys: 'hr', 'hrv_sdnn', 'resp_rate', 'physical_effort'
    Each value is list of (datetime, float) tuples, sorted by time.
    """
    print("  Parsing export.xml (may take 30-60s for 116 MB) ...")
    records: dict[str, list] = {
        "hr": [], "hrv_sdnn": [], "resp_rate": [], "physical_effort": []
    }
    type_map = {
        "HKQuantityTypeIdentifierHeartRate"                 : "hr",
        "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"  : "hrv_sdnn",
        "HKQuantityTypeIdentifierRespiratoryRate"           : "resp_rate",
        "HKQuantityTypeIdentifierPhysicalEffort"            : "physical_effort",
    }
    count = 0
    for event, elem in ET.iterparse(str(xml_path), events=["end"]):
        if elem.tag == "Record":
            rtype = elem.attrib.get("type", "")
            key   = type_map.get(rtype)
            if key:
                try:
                    val = float(elem.attrib["value"])
                    dt  = datetime.strptime(elem.attrib["startDate"][:19],
                                            "%Y-%m-%d %H:%M:%S")
                    records[key].append((dt, val))
                except Exception:
                    pass
            elem.clear()
            count += 1
            if count % 100_000 == 0:
                print(f"    {count:,} records processed...")

    for key in records:
        records[key].sort(key=lambda x: x[0])
    print(f"  Done. Parsed {count:,} total records.")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_vector(hrv_feats: dict,
                         resp_rate: float | None,
                         physical_effort: float | None) -> np.ndarray:
    """
    Assemble the 16-element feature vector.
    EDA features always imputed. ACC partially from Apple Watch signals.
    """
    feat = {**SAFE_MEDIANS}               # start with all safe medians
    feat.update(hrv_feats)                # overwrite HRV with computed values
    if resp_rate is not None:
        feat["acc_breathing_rate"] = resp_rate
    if physical_effort is not None:
        # PhysicalEffort (MET-like) -> rough proxy for motion magnitude
        # WESAD acc_mean_mag safe median ~63 (arbitrary units)
        # Scale: effort 1.0 ~ resting (63), effort 5.0 ~ active (100)
        feat["acc_mean_mag"] = 50.0 + physical_effort * 10.0
        feat["acc_std_mag"]  = 1.0 + physical_effort * 0.5

    return np.array([feat[n] for n in FEATURE_NAMES], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ECG mode
# ─────────────────────────────────────────────────────────────────────────────

def run_ecg_mode(model) -> None:
    ecg_dir = AW_DIR / "electrocardiograms"
    ecg_files = sorted(ecg_dir.glob("ecg_*.csv"))
    if not ecg_files:
        print("  No ECG files found.")
        return

    print(f"\n{'='*60}")
    print("  ECG MODE  (true R-peak HRV from 30s Lead-I ECG)")
    print(f"{'='*60}")
    print(f"  {'File':<30}  {'Pred':>8}  {'Conf':>8}  {'HRV RMSSD':>10}  {'HR est':>8}")
    print("  " + "-"*68)

    for ecg_path in ecg_files:
        lines  = open(ecg_path).readlines()
        # skip header lines until pure numeric
        data_start = next(i for i, l in enumerate(lines)
                          if l.strip().lstrip("-").replace(".", "").isdigit())
        ecg = np.array([float(l.strip()) for l in lines[data_start:]
                        if l.strip().lstrip("-").replace(".", "").isdigit()],
                       dtype=np.float64)

        hrv    = _hrv_from_ecg(ecg, fs=512.0)
        fvec   = build_feature_vector(hrv, resp_rate=None, physical_effort=None)
        prob   = model.predict_proba(fvec.reshape(1, -1))[0, 1]
        spw    = 7.67
        thr    = 1.0 / (spw + 1.0)
        pred   = "DISTRESS" if prob >= thr else "safe"
        hr_est = 60.0 / hrv["hrv_mean_rr"] if hrv["hrv_mean_rr"] > 0 else 0

        print(f"  {ecg_path.stem:<30}  {pred:>8}  {prob:>7.3f}  "
              f"{hrv['hrv_rmssd']:>10.4f}  {hr_est:>7.1f} bpm")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# HR continuous mode
# ─────────────────────────────────────────────────────────────────────────────

def run_hr_mode(records: dict, model) -> None:
    hr_data   = records["hr"]
    resp_data = records["resp_rate"]
    eff_data  = records["physical_effort"]

    if not hr_data:
        print("  No HR data found.")
        return

    print(f"\n{'='*60}")
    print("  HR CONTINUOUS MODE  (30s windows, estimated RR from HR)")
    print(f"{'='*60}")

    # Build lookup: timestamp -> nearest resp_rate, physical_effort
    def nearest_val(data: list, target: datetime, max_gap_min: int = 5) -> float | None:
        if not data:
            return None
        best = min(data, key=lambda x: abs((x[0] - target).total_seconds()))
        if abs((best[0] - target).total_seconds()) <= max_gap_min * 60:
            return best[1]
        return None

    # Slide 30s windows (5s step) over the HR time series
    hr_times  = np.array([t.timestamp() for t, _ in hr_data])
    hr_vals   = np.array([v for _, v in hr_data])
    t_start   = hr_times[0]
    t_end     = hr_times[-1]
    win_sec   = 30.0
    step_sec  = 300.0   # 5-minute step for summary (not every 5s — too many)

    windows = []
    t = t_start
    while t + win_sec <= t_end:
        mask = (hr_times >= t) & (hr_times < t + win_sec)
        if mask.sum() >= 2:
            win_hr  = hr_vals[mask]
            win_dt  = datetime.fromtimestamp(t + win_sec / 2)
            resp    = nearest_val(resp_data, win_dt)
            effort  = nearest_val(eff_data, win_dt)
            hrv     = _hrv_from_hr_window(win_hr)
            fvec    = build_feature_vector(hrv, resp, effort)
            prob    = float(model.predict_proba(fvec.reshape(1, -1))[0, 1])
            windows.append((win_dt, prob, hrv["hrv_mean_rr"], hrv["hrv_rmssd"]))
        t += step_sec

    if not windows:
        print("  Not enough HR data for windowed prediction.")
        return

    dts   = [w[0] for w in windows]
    probs = np.array([w[1] for w in windows])
    spw   = 7.67
    thr   = 1.0 / (spw + 1.0)
    preds = (probs >= thr).astype(int)

    n_distress = int(preds.sum())
    n_safe     = int((~preds.astype(bool)).sum())
    print(f"\n  Total windows  : {len(windows):,}")
    print(f"  Date range     : {dts[0].date()} to {dts[-1].date()}")
    print(f"  Safe windows   : {n_safe:,}  ({100*n_safe/len(windows):.1f}%)")
    print(f"  Distress flags : {n_distress:,}  ({100*n_distress/len(windows):.1f}%)")
    print(f"  Mean confidence: {probs.mean():.3f}")
    print(f"  Threshold used : {thr:.3f}")

    # Print highest-stress moments
    top_idx = np.argsort(probs)[::-1][:10]
    print(f"\n  Top 10 highest-stress moments:")
    print(f"  {'Datetime':<22}  {'Conf':>7}  {'Pred':>9}  {'HR est':>8}  {'RMSSD':>8}")
    print("  " + "-"*60)
    for i in top_idx:
        dt, prob, mean_rr, rmssd = windows[i]
        hr_est = 60.0 / mean_rr if mean_rr > 0 else 0
        pred   = "DISTRESS" if prob >= thr else "safe"
        print(f"  {str(dt)[:19]:<22}  {prob:>7.3f}  {pred:>9}  {hr_est:>7.1f}  {rmssd:>8.4f}")

    # Plot timeline
    _plot_timeline(dts, probs, thr)


def _plot_timeline(dts, probs: np.ndarray, thr: float) -> None:
    fig, ax = plt.subplots(figsize=(15, 4))
    colors = ["#e74c3c" if p >= thr else "#2980b9" for p in probs]
    ax.scatter(dts, probs, c=colors, s=8, alpha=0.6, linewidths=0)
    ax.axhline(thr, color="darkorange", linestyle="--", linewidth=1.5,
               label=f"Threshold = {thr:.3f}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Distress probability", fontsize=11)
    ax.set_title("Guardian Drone — Apple Watch Stress Timeline\n"
                 "Red = distress flag  |  Blue = safe  |  "
                 "NOTE: EDA imputed (no GSR sensor)", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = DOCS_DIR / "apple_watch_timeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Timeline plot saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "="*60)
    print("  Guardian Drone — Apple Watch Adapter")
    print("  Baseline XGBoost Predictor (Phase 1.5)")
    print("="*60)
    print()
    print("  IMPORTANT CAVEAT:")
    print("  Apple Watch has NO EDA/GSR sensor.")
    print("  The 6 EDA features are imputed with WESAD safe-class medians.")
    print("  'eda_tonic_mean' was the #1 distress predictor -> accuracy is reduced.")
    print("  Predictions reflect HRV + activity patterns only.")
    print()

    # load model
    model_path = MODELS_DIR / "baseline_xgb.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"  Model loaded: {model_path.name}")

    # ECG mode
    run_ecg_mode(model)

    # HR continuous mode
    xml_path = AW_DIR / "export.xml"
    records  = parse_health_xml(xml_path)
    run_hr_mode(records, model)

    print("\n  Done. Check docs/apple_watch_timeline.png for the full timeline.")


if __name__ == "__main__":
    main()
