"""
End-to-end pipeline test with synthetic scenarios.

Two scenarios are simulated second-by-second through the full stack:
  LSTM Danger Classifier  ->  ContextualGate  ->  SOS decision

Scenario 1 — Exercise (5 minutes, running):
  Elevated movement, elevated breathing, moderate EDA (sweat not fear).
  Expected: model scores stay below 70, gate NEVER fires SOS.

Scenario 2 — Genuine Distress (5 minutes):
  Phase 1 (0-60s):   Walking home, baseline physiology.
  Phase 2 (60-120s): Escalating fear (being followed).
  Phase 3 (120s+):   Full distress — EDA spikes, HRV drops, hyperventilation.
  Expected: gate fires SOS ~15-20s after Phase 3 begins.

Run:
    C:/Users/akhil/anaconda3/envs/LLM_GPU/python.exe simulation/scenario_test.py
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.contextual_gate import ContextualGate, check_multi_signal

MODELS_DIR = ROOT / "biometric_ml" / "models"
DOCS_DIR   = ROOT / "docs"

FEATURE_NAMES = [
    "hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hrv_lf_hf", "hrv_mean_rr", "hrv_std_rr",
    "eda_mean", "eda_std", "eda_phasic_peaks", "eda_phasic_amplitude", "eda_slope",
    "eda_tonic_mean", "acc_mean_mag", "acc_std_mag", "acc_spectral_entropy",
    "acc_breathing_rate",
]

SEQ_LEN    = 8
N_FEATURES = 16
WINDOW_SEC = 5      # one window every 5 seconds
RNG        = np.random.default_rng(42)


# =============================================================================
# LSTM MODEL (must match biometric_ml.lstm_classifier.DangerLSTM)
# =============================================================================

class DangerLSTM(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, n_layers,
                            dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2)
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


# =============================================================================
# PROTOTYPE FEATURE VECTORS (in raw WESAD feature space)
# Values derived from WESAD feature distribution analysis (all_subjects.npz).
# =============================================================================

# ── Resting baseline (walking home, calm) ─────────────────────────────────────
BASELINE = np.array([
    0.30,  0.23,  0.71,  0.46,  0.83,  0.23,   # HRV (safe medians from WESAD)
    0.63,  0.012, 3.0,   0.010, -0.0002, 0.63,  # EDA (safe medians)
    63.4,  1.36,  0.92,  15.0,                  # ACC (safe medians)
], dtype=np.float32)

# ── Escalating fear (noticing being followed) ─────────────────────────────────
ESCALATING = np.array([
    0.18,  0.15,  0.35,  1.80,  0.74,  0.15,   # HRV: rmssd dropping, lf_hf rising
    1.20,  0.030, 4.0,   0.025, 0.0001, 1.20,  # EDA: conductance rising
    65.0,  2.0,   0.89,  18.0,                  # ACC: slight movement increase
], dtype=np.float32)

# ── Full distress (acute fear — attack / threat) ──────────────────────────────
# EDA matches WESAD distress median (model trained on this).
# HRV shows acute fear sympathetic response.
# Breathing shows hyperventilation.
DISTRESS = np.array([
    0.12,  0.09,  0.12,  3.20,  0.68,  0.09,   # HRV: low rmssd, high lf_hf, fast HR
    2.10,  0.052, 6.0,   0.040, 0.0002, 2.10,  # EDA: matches WESAD distress exactly
    58.0,  3.10,  0.86,  22.0,                  # ACC: frozen + hyperventilation
], dtype=np.float32)

# ── Exercise (running) ────────────────────────────────────────────────────────
# EDA slightly elevated (thermoregulatory sweat, NOT fear-sweat level).
# HRV reflects elevated heart rate from aerobic effort.
# ACC shows heavy movement + elevated breathing.
EXERCISE = np.array([
    0.24,  0.19,  0.45,  1.00,  0.62,  0.19,   # HRV: moderately lower (HR ~97 bpm)
    0.82,  0.022, 2.0,   0.013, 0.00003, 0.82, # EDA: mild sweat (below distress level)
    95.0,  4.80,  0.72,  25.0,                  # ACC: high movement, elevated breathing
], dtype=np.float32)


def _add_noise(vec: np.ndarray, scale: float = 0.03) -> np.ndarray:
    """Add small physiological noise so the trace looks like real sensor data."""
    noise = RNG.standard_normal(len(vec)) * scale * np.abs(vec)
    return (vec + noise).astype(np.float32)


# =============================================================================
# BUILD SCENARIO WINDOW SEQUENCES
# =============================================================================

def build_exercise_windows(n_total: int = 60) -> np.ndarray:
    """
    0 - 12s  (3 windows):  Stretching / getting ready — baseline
    12 - 35s (5 windows):  Warm-up jog — gradual increase
    35s+     (remaining):  Full run — sustained exercise features
    """
    windows = []
    for i in range(n_total):
        if i < 3:
            w = _add_noise(BASELINE)
        elif i < 8:
            alpha = (i - 3) / 5
            w = _add_noise((1 - alpha) * BASELINE + alpha * EXERCISE)
        else:
            w = _add_noise(EXERCISE)
        windows.append(w)
    return np.array(windows, dtype=np.float32)


def build_distress_windows(n_total: int = 60) -> np.ndarray:
    """
    0 - 60s  (12 windows): Walking home — baseline
    60 - 120s (12 windows): Escalation — noticing threat
    120s+    (remaining):  Full distress — attack / chase
    """
    windows = []
    for i in range(n_total):
        if i < 12:
            w = _add_noise(BASELINE)
        elif i < 24:
            alpha = (i - 12) / 12
            w = _add_noise((1 - alpha) * BASELINE + alpha * ESCALATING)
        else:
            alpha = min(1.0, (i - 24) / 8)
            w = _add_noise((1 - alpha) * ESCALATING + alpha * DISTRESS)
        windows.append(w)
    return np.array(windows, dtype=np.float32)


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def run_simulation(
    model:    DangerLSTM,
    scaler:   StandardScaler,
    gate:     ContextualGate,
    windows:  np.ndarray,
    label:    str,
) -> list[dict]:
    """
    Feed windows one at a time to simulate a streaming watch session.
    Returns one result dict per window once we have >= 8 for the first sequence.
    """
    model.eval()
    seq_buf = deque(maxlen=SEQ_LEN)
    results = []

    with torch.no_grad():
        for i, raw_w in enumerate(windows):
            t = float(i * WINDOW_SEC)

            norm_w = scaler.transform(raw_w.reshape(1, -1))[0].astype(np.float32)
            seq_buf.append(norm_w)

            if len(seq_buf) < SEQ_LEN:
                continue

            seq   = np.array(seq_buf, dtype=np.float32)[np.newaxis]    # (1, 8, 16)
            logit = model(torch.from_numpy(seq))
            prob  = float(torch.softmax(logit, dim=-1)[0, 1].item())
            raw_score = prob * 100.0

            agree = check_multi_signal(raw_w)
            dec   = gate.process(raw_score, multi_signal_agree=agree, now=t)

            results.append({
                "t":           t,
                "raw_score":   raw_score,
                "smoothed":    dec.smoothed_score,
                "agree":       agree,
                "blocked_by":  dec.blocked_by,
                "sos":         dec.should_sos,
            })

    return results


# =============================================================================
# REPORT + PLOT
# =============================================================================

def print_report(results: list[dict], label: str) -> None:
    sos_times = [r["t"] for r in results if r["sos"]]
    first_sos = sos_times[0] if sos_times else None

    print(f"\n  {'=' * 60}")
    print(f"  Scenario: {label}")
    print(f"  {'=' * 60}")
    print(f"  {'Time':>6}  {'RawScore':>9}  {'Smoothed':>9}  {'Agree':>6}  {'Gate Status':<22}")
    print(f"  {'-' * 60}")
    for r in results:
        sos_flag = "  <<< SOS FIRED" if r["sos"] else ""
        print(f"  {r['t']:>6.0f}s  {r['raw_score']:>9.1f}  {r['smoothed']:>9.1f}"
              f"  {'YES' if r['agree'] else 'NO':>6}  {r['blocked_by']:<22}{sos_flag}")
    print(f"  {'-' * 60}")
    if first_sos:
        print(f"  RESULT: SOS fired at t={first_sos:.0f}s")
    else:
        print(f"  RESULT: NO SOS triggered (gate held)")
    print()


def plot_scenarios(
    exercise_results: list[dict],
    distress_results: list[dict],
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    for ax, results, title, color in zip(
        axes,
        [exercise_results, distress_results],
        ["Scenario 1 — Exercise (Running)", "Scenario 2 — Genuine Distress (Attack / Threat)"],
        ["#2196F3", "#F44336"],
    ):
        times    = [r["t"] for r in results]
        raw_sc   = [r["raw_score"] for r in results]
        smooth   = [r["smoothed"]  for r in results]
        agree    = [r["agree"]     for r in results]
        sos_fire = [r["t"] for r in results if r["sos"]]

        # background zones by agree status
        for i in range(len(times)):
            bg = "#FFF3E0" if agree[i] else "#E8F5E9"
            x0 = times[i] - WINDOW_SEC / 2
            x1 = times[i] + WINDOW_SEC / 2
            ax.axvspan(x0, x1, alpha=0.25, color=bg, linewidth=0)

        ax.plot(times, raw_sc, color=color, alpha=0.4, linewidth=1.2, label="Raw LSTM score")
        ax.plot(times, smooth, color=color, linewidth=2.2, label="Smoothed (10-window avg)")
        ax.axhline(70, color="#E53935", ls="--", lw=1.4, label="SOS threshold (70)")
        ax.axhline(50, color="#FB8C00", ls=":", lw=1.0, alpha=0.6, label="Caution (50)")

        for t_sos in sos_fire[:1]:    # mark only first SOS
            ax.axvline(t_sos, color="#B71C1C", lw=2.5, ls="-", label=f"SOS fired t={t_sos:.0f}s")
            ax.annotate(f"SOS FIRED\nt={t_sos:.0f}s",
                        xy=(t_sos, 70), xytext=(t_sos + 8, 82),
                        fontsize=10, color="#B71C1C", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#B71C1C"))

        if not sos_fire:
            ax.text(0.98, 0.92, "NO SOS — Gate held",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=11, color="#2E7D32", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9"))

        ax.set_ylim(0, 105)
        ax.set_xlim(min(times) - 2, max(times) + 2)
        ax.set_ylabel("Danger Score (0-100)", fontsize=11)
        ax.set_xlabel("Time (seconds)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Guardian Drone — End-to-End Pipeline Test\n"
                 "LSTM + Contextual Gate | Synthetic Physiological Data",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = DOCS_DIR / "scenario_test.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved -> {out}")
    return out


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "=" * 64)
    print("  Guardian Drone — End-to-End Scenario Test")
    print("=" * 64)

    # ── Load WESAD features to fit scaler ─────────────────────────────────────
    print("\n  Loading WESAD features to fit normalization scaler ...")
    processed = ROOT / "data" / "processed" / "all_subjects.npz"
    d = np.load(processed, allow_pickle=True)
    X_all = d["X"].astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(X_all)
    print(f"  Scaler fitted on {len(X_all):,} windows from 15 subjects.")

    # ── Load LSTM model ───────────────────────────────────────────────────────
    print("\n  Loading DangerLSTM ...")
    state_dict = torch.load(MODELS_DIR / "best_danger_model.pt",
                            map_location="cpu", weights_only=True)
    model = DangerLSTM()
    model.load_state_dict(state_dict)
    model.eval()
    print("  Model loaded.")

    # ── Print prototype feature values ────────────────────────────────────────
    print("\n  Prototype feature values (raw WESAD space):")
    print(f"  {'Feature':<22} {'Baseline':>10} {'Exercise':>10} {'Escalate':>10} {'Distress':>10}")
    print("  " + "-" * 64)
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:<22} {BASELINE[i]:>10.4f} {EXERCISE[i]:>10.4f}"
              f" {ESCALATING[i]:>10.4f} {DISTRESS[i]:>10.4f}")

    # ── Check multi-signal agreement for each prototype ───────────────────────
    print("\n  Multi-signal agreement check:")
    for label, vec in [("Baseline", BASELINE), ("Exercise", EXERCISE),
                       ("Escalating", ESCALATING), ("Distress", DISTRESS)]:
        agree = check_multi_signal(vec)
        print(f"    {label:<12}: {'AGREE (>=2/3 signals)' if agree else 'NO AGREE (< 2/3 signals)'}")

    # ── Build windows ─────────────────────────────────────────────────────────
    print("\n  Building synthetic scenario windows (60 windows x 5s = 5 min each) ...")
    ex_windows   = build_exercise_windows(n_total=60)
    dist_windows = build_distress_windows(n_total=60)

    # ── Run simulations ───────────────────────────────────────────────────────
    print("\n  Running Exercise scenario ...")
    ex_gate   = ContextualGate()
    ex_res    = run_simulation(model, scaler, ex_gate,   ex_windows,   "Exercise")

    print("  Running Distress scenario ...")
    dist_gate = ContextualGate()
    dist_res  = run_simulation(model, scaler, dist_gate, dist_windows, "Genuine Distress")

    # ── Report ────────────────────────────────────────────────────────────────
    print_report(ex_res,   "Exercise (Running)")
    print_report(dist_res, "Genuine Distress (Attack / Threat)")

    # ── Summary ───────────────────────────────────────────────────────────────
    ex_sos   = any(r["sos"] for r in ex_res)
    dist_sos = any(r["sos"] for r in dist_res)

    print("  " + "=" * 64)
    print("  FINAL RESULT")
    print("  " + "=" * 64)
    print(f"  Exercise -> SOS fired? {'YES (FALSE POSITIVE)' if ex_sos else 'NO -- CORRECT'}")
    print(f"  Distress -> SOS fired? {'YES -- CORRECT' if dist_sos else 'NO (MISSED THREAT)'}")

    ex_ok   = not ex_sos
    dist_ok = dist_sos
    overall = "PASS" if (ex_ok and dist_ok) else "FAIL"
    print(f"\n  Overall pipeline test: {overall}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\n  Generating plot ...")
    plot_scenarios(ex_res, dist_res)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
