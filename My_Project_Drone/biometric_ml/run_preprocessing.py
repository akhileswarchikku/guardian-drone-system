"""
Phase 1.3 — Run preprocessing pipeline on all 15 WESAD subjects.

What this script does:
  1. Loads each subject's .pkl file
  2. Slides a 30-second window (5-second step) over BVP, EDA, ACC
  3. Extracts 16 features per window
  4. Aligns binary labels (stress=1, safe=0) via majority vote
  5. Saves per-subject arrays to data/processed/
  6. Saves a combined dataset (all subjects stacked) to data/processed/all_subjects.npz
  7. Prints a dataset summary

Run with:
    conda activate LLM
    cd D:/My_Project_Drone
    python -m biometric_ml.run_preprocessing
"""

from pathlib import Path
import numpy as np
from tqdm import tqdm

from biometric_ml.data_loader import load_subject, SUBJECT_IDS
from biometric_ml.preprocessing import build_windows, FEATURE_NAMES

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def run_all_subjects(force: bool = False) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Process all 15 subjects.

    Parameters
    ----------
    force : bool
        If False, skip subjects whose .npz already exists (resume mode).

    Returns
    -------
    X_all : (N_total_windows, 16)  float32
    y_all : (N_total_windows,)     int32
    subject_ids_per_window : list of int, length N_total_windows
    """
    all_X, all_y, all_ids = [], [], []

    print("\n" + "=" * 60)
    print("  Guardian Drone System — Phase 1 Preprocessing")
    print("  Running on all 15 WESAD subjects")
    print("=" * 60)
    print(f"  Window : {30}s  |  Step : {5}s  |  Features : {len(FEATURE_NAMES)}")
    print("=" * 60 + "\n")

    for sid in tqdm(SUBJECT_IDS, desc="Subjects", unit="subj"):
        out_path = PROCESSED_DIR / f"S{sid}_features.npz"

        if out_path.exists() and not force:
            data = np.load(out_path)
            X, y = data["X"], data["y"]
            n_distress = int(np.sum(y == 1))
            n_safe     = int(np.sum(y == 0))
            print(f"  S{sid:2d}: loaded from cache — "
                  f"{len(y):4d} windows  distress={n_distress}  safe={n_safe}")
        else:
            subject_data = load_subject(sid)
            X, y = build_windows(subject_data, sid)
            np.savez_compressed(out_path, X=X, y=y)

        all_X.append(X)
        all_y.append(y)
        all_ids.extend([sid] * len(y))

    X_all = np.vstack(all_X).astype(np.float32)
    y_all = np.concatenate(all_y).astype(np.int32)

    # save combined dataset
    combined_path = PROCESSED_DIR / "all_subjects.npz"
    np.savez_compressed(
        combined_path,
        X=X_all,
        y=y_all,
        subject_ids=np.array(all_ids, dtype=np.int32),
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"\n  Combined dataset saved -> {combined_path}")
    return X_all, y_all, all_ids


def print_dataset_summary(X: np.ndarray, y: np.ndarray, subject_ids: list) -> None:
    """Print class balance and per-feature statistics."""
    n_total    = len(y)
    n_distress = int(np.sum(y == 1))
    n_safe     = int(np.sum(y == 0))

    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total windows   : {n_total:,}")
    print(f"  Distress (y=1)  : {n_distress:,}  ({100*n_distress/n_total:.1f}%)")
    print(f"  Safe     (y=0)  : {n_safe:,}  ({100*n_safe/n_total:.1f}%)")
    print(f"  Subjects        : {len(set(subject_ids))}")
    print(f"  Features        : {X.shape[1]}")
    print()

    print(f"  {'Feature':<28}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    print("  " + "-" * 64)
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        print(f"  {name:<28}  {col.mean():8.4f}  {col.std():8.4f}  "
              f"{col.min():8.4f}  {col.max():8.4f}")

    print("\n  Checklist target: LSTM F1 > 0.83  |  XGBoost F1 > 0.78")
    print("  scale_pos_weight for XGBoost ≈ "
          f"{n_safe / max(n_distress, 1):.2f}  (use ~3.0 as per checklist)")


if __name__ == "__main__":
    X_all, y_all, ids = run_all_subjects(force=False)
    print_dataset_summary(X_all, y_all, ids)
