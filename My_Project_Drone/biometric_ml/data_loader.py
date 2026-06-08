"""
Phase 1.2 — WESAD Dataset Loader
Loads subject .pkl files, verifies integrity, and plots sample signal traces.

WESAD wrist sensor (Empatica E4) sampling rates:
  BVP  : 64  Hz  (Blood Volume Pulse — source for HRV)
  EDA  : 4   Hz  (Electrodermal Activity / GSR)
  ACC  : 32  Hz  (Accelerometer — 3-axis: x, y, z)
  TEMP : 4   Hz  (Skin temperature — not used in this project)

Labels (at 700 Hz):
  0 = not defined / transient
  1 = baseline
  2 = stress         ← distress=1 in our binary model
  3 = amusement      ← safe=0
  4 = meditation     ← safe=0 (only some subjects)

Subjects: S2–S17 (S1 and S12 excluded from the official dataset).
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves to file, no popup
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw" / "WESAD"
DOCS_DIR = ROOT / "docs"

# ── constants ───────────────────────────────────────────────────────────────
SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

LABEL_NAMES = {
    0: "not_defined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}

FS = {
    "BVP":   64,
    "EDA":   4,
    "ACC":   32,
    "TEMP":  4,
    "label": 700,
}


# ── core loaders ────────────────────────────────────────────────────────────

def load_subject(subject_id: int) -> dict:
    """Load one subject's .pkl and return the raw data dict."""
    pkl_path = DATA_DIR / f"S{subject_id}" / f"S{subject_id}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def get_wrist_signals(data: dict) -> dict:
    """Return the wrist sensor dict: keys BVP, EDA, ACC, TEMP."""
    return data["signal"]["wrist"]


def get_labels(data: dict) -> np.ndarray:
    """Return flattened label array at 700 Hz."""
    return data["label"].flatten()


# ── inspection helpers ───────────────────────────────────────────────────────

def print_subject_info(subject_id: int) -> None:
    """Print signal shapes and label-count breakdown for one subject."""
    data   = load_subject(subject_id)
    wrist  = get_wrist_signals(data)
    labels = get_labels(data)

    print(f"\n{'='*55}")
    print(f"  Subject S{subject_id}")
    print(f"{'='*55}")
    print("  Wrist signals:")
    for key, arr in wrist.items():
        fs  = FS.get(key, "?")
        dur = arr.shape[0] / fs if isinstance(fs, int) else "?"
        print(f"    {key:5s}: shape={arr.shape}  fs={fs} Hz  "
              f"duration={dur:.1f}s" if isinstance(dur, float) else
              f"    {key:5s}: shape={arr.shape}  fs={fs} Hz")

    print(f"\n  Labels: shape={labels.shape}  fs={FS['label']} Hz")
    print("  Label breakdown:")
    for label_id, name in LABEL_NAMES.items():
        count = int(np.sum(labels == label_id))
        if count:
            dur = count / FS["label"]
            print(f"    {label_id} ({name:12s}): {count:7d} samples  "
                  f"({dur:5.1f}s)")


def verify_all_subjects() -> bool:
    """
    Load all 15 subjects and verify the three signals + stress labels exist.
    Prints a per-subject result and returns True only if all pass.
    """
    print("\nVerifying all WESAD subjects …\n")
    ok, failed = [], []

    for sid in SUBJECT_IDS:
        try:
            data   = load_subject(sid)
            wrist  = get_wrist_signals(data)
            labels = get_labels(data)

            assert "BVP" in wrist,               "BVP missing"
            assert "EDA" in wrist,               "EDA missing"
            assert "ACC" in wrist,               "ACC missing"
            assert wrist["BVP"].size > 0,        "BVP empty"
            assert wrist["EDA"].size > 0,        "EDA empty"
            assert wrist["ACC"].size > 0,        "ACC empty"
            stress_n = int(np.sum(labels == 2))
            assert stress_n > 0,                 "no stress labels"

            bvp_dur = wrist["BVP"].shape[0] / FS["BVP"]
            print(f"  S{sid:2d}: OK — BVP {bvp_dur:.0f}s  "
                  f"stress_samples={stress_n}")
            ok.append(sid)

        except Exception as exc:
            print(f"  S{sid:2d}: FAILED — {exc}")
            failed.append(sid)

    print(f"\nResult: {len(ok)}/{len(SUBJECT_IDS)} subjects OK")
    if failed:
        print(f"Failed: {failed}")
    return len(failed) == 0


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_sample_traces(
    subject_ids: tuple = (2, 5, 11),
    save_path: Path | None = None,
) -> None:
    """
    Plot 60 seconds of BVP, EDA, and ACC-magnitude from the stress segment
    for each subject in subject_ids.  Saves to docs/sample_traces.png.
    """
    n_rows = len(subject_ids)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows))
    fig.suptitle(
        "WESAD Wrist Sensor — Stress-Segment Sample Traces\n"
        "(60 seconds per subject)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for row, sid in enumerate(subject_ids):
        data   = load_subject(sid)
        wrist  = get_wrist_signals(data)
        labels = get_labels(data)

        # find start of first stress segment
        stress_idxs = np.where(labels == 2)[0]
        if len(stress_idxs) == 0:
            print(f"  S{sid}: no stress labels — skipping plot")
            continue
        t_start_sec = stress_idxs[0] / FS["label"]

        win_sec = 60
        bvp_s = int(t_start_sec * FS["BVP"])
        eda_s = int(t_start_sec * FS["EDA"])
        acc_s = int(t_start_sec * FS["ACC"])

        bvp = wrist["BVP"].flatten()[bvp_s: bvp_s + win_sec * FS["BVP"]]
        eda = wrist["EDA"].flatten()[eda_s: eda_s + win_sec * FS["EDA"]]
        acc = wrist["ACC"][acc_s: acc_s + win_sec * FS["ACC"]]
        acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))

        t_bvp = np.arange(len(bvp)) / FS["BVP"]
        t_eda = np.arange(len(eda)) / FS["EDA"]
        t_acc = np.arange(len(acc_mag)) / FS["ACC"]

        # BVP
        ax = axes[row, 0] if n_rows > 1 else axes[0]
        ax.plot(t_bvp, bvp, color="steelblue", linewidth=0.6)
        ax.set_title(f"S{sid} — BVP (HRV source)", fontsize=10)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("BVP (a.u.)")
        ax.set_xlim(0, win_sec)

        # EDA
        ax = axes[row, 1] if n_rows > 1 else axes[1]
        ax.plot(t_eda, eda, color="darkorange", linewidth=1.0)
        ax.set_title(f"S{sid} — EDA / GSR", fontsize=10)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("EDA (µS)")
        ax.set_xlim(0, win_sec)

        # ACC magnitude
        ax = axes[row, 2] if n_rows > 1 else axes[2]
        ax.plot(t_acc, acc_mag, color="forestgreen", linewidth=0.7)
        ax.set_title(f"S{sid} — ACC Magnitude", fontsize=10)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Magnitude (g)")
        ax.set_xlim(0, win_sec)

    plt.tight_layout()

    out = save_path or DOCS_DIR / "sample_traces.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    plt.close()


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1 — inspect one subject
    print_subject_info(2)

    # Step 2 — verify all 15
    all_ok = verify_all_subjects()

    # Step 3 — plot sample traces for 3 subjects
    if all_ok:
        plot_sample_traces(subject_ids=(2, 5, 11))
