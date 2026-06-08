"""
Phase 1.5 — XGBoost Baseline Classifier

Purpose:
  Validate the 16-feature preprocessing pipeline before building the LSTM.
  XGBoost trains in minutes and SHAP values reveal exactly which features
  predict distress. If F1 < 0.78, the preprocessing has a bug — fix it here,
  not after 30 epochs of LSTM training.

GPU:
  Uses XGBoost GPU acceleration (device='cuda') with NVIDIA GeForce MX150.
  Automatically falls back to CPU if CUDA is unavailable.

Evaluation:
  Leave-One-Subject-Out Cross-Validation (LOSO-CV) — standard for biometric studies.
  Train on 14 subjects, test on 1, repeat for all 15 subjects.

Target: mean F1 > 0.78 on distress class across all 15 LOSO folds.

Run:
    conda activate LLM_GPU
    cd D:/My_Project_Drone
    python -m biometric_ml.xgboost_classifier
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
import shap
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm

from biometric_ml.preprocessing import FEATURE_NAMES, N_FEATURES
from biometric_ml.data_loader import SUBJECT_IDS

# ── paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "biometric_ml" / "models"
DOCS_DIR      = ROOT / "docs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# GPU DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_device() -> str:
    """Return 'cuda' if XGBoost can use GPU, else 'cpu'."""
    try:
        probe = xgb.XGBClassifier(n_estimators=1, device="cuda", verbosity=0)
        probe.fit(
            np.random.rand(20, 4).astype(np.float32),
            np.array([0] * 10 + [1] * 10, dtype=np.int32),
        )
        print("  GPU detected (CUDA) — XGBoost will use GPU acceleration")
        return "cuda"
    except Exception as e:
        print(f"  GPU not available ({e}) — using CPU")
        return "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed feature matrix from data/processed/all_subjects.npz.
    Runs the preprocessing pipeline first if file is missing.

    Returns
    -------
    X      : (N, 16) float32  — feature windows
    y      : (N,)    int32    — binary labels (1=distress, 0=safe)
    groups : (N,)    int32    — subject ID per window (for LOSO-CV)
    """
    combined = PROCESSED_DIR / "all_subjects.npz"

    if not combined.exists():
        print("  Preprocessed data not found — running preprocessing now …")
        from biometric_ml.run_preprocessing import run_all_subjects
        X_all, y_all, ids = run_all_subjects()
        return X_all, y_all, np.array(ids, dtype=np.int32)

    print(f"  Loading from {combined.name} …")
    data   = np.load(combined, allow_pickle=True)
    X      = data["X"].astype(np.float32)
    y      = data["y"].astype(np.int32)
    groups = data["subject_ids"].astype(np.int32)

    n_dist = int(np.sum(y == 1))
    n_safe = int(np.sum(y == 0))
    print(f"  {len(y):,} windows  |  distress={n_dist:,}  safe={n_safe:,}  "
          f"ratio=1:{n_safe//max(n_dist,1):.1f}")
    return X, y, groups


# ═══════════════════════════════════════════════════════════════════════════
# MODEL BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_model(scale_pos_weight: float, device: str) -> xgb.XGBClassifier:
    """
    XGBoost with GPU acceleration.
    scale_pos_weight compensates for class imbalance (n_safe / n_distress).
    """
    return xgb.XGBClassifier(
        n_estimators          = 500,
        max_depth             = 6,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 5,
        gamma                 = 0.1,
        scale_pos_weight      = scale_pos_weight,
        tree_method           = "hist",     # required for GPU in XGBoost 2.x+
        device                = device,
        eval_metric           = "logloss",
        early_stopping_rounds = 20,
        random_state          = 42,
        verbosity             = 0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# LOSO-CV
# ═══════════════════════════════════════════════════════════════════════════

def run_loso_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    device: str,
) -> dict:
    """
    Leave-One-Subject-Out Cross-Validation.
    Train on 14 subjects → test on held-out 1. Repeat 15 times.
    """
    logo  = LeaveOneGroupOut()
    folds = list(logo.split(X, y, groups))

    results = {
        "subject_ids": [],
        "f1":          [],
        "precision":   [],
        "recall":      [],
        "roc_auc":     [],
        "y_true_all":  [],
        "y_pred_all":  [],
        "y_prob_all":  [],
    }

    print(f"\n  Running LOSO-CV  ({len(folds)} folds) …\n")

    for train_idx, test_idx in tqdm(folds, desc="  LOSO-CV", unit="fold"):
        sid = int(groups[test_idx[0]])

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        n_safe_tr = int(np.sum(y_tr == 0))
        n_dist_tr = int(np.sum(y_tr == 1))
        spw       = n_safe_tr / max(n_dist_tr, 1)

        model = build_model(scale_pos_weight=spw, device=device)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=False,
        )

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        f1  = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
        pre = precision_score(y_te, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_te, y_pred, pos_label=1, zero_division=0)
        auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.0

        results["subject_ids"].append(sid)
        results["f1"].append(f1)
        results["precision"].append(pre)
        results["recall"].append(rec)
        results["roc_auc"].append(auc)
        results["y_true_all"].extend(y_te.tolist())
        results["y_pred_all"].extend(y_pred.tolist())
        results["y_prob_all"].extend(y_prob.tolist())

        tqdm.write(
            f"    S{sid:2d}: F1={f1:.3f}  Prec={pre:.3f}  Rec={rec:.3f}  "
            f"AUC={auc:.3f}  "
            f"[distress={int(np.sum(y_te==1))}  safe={int(np.sum(y_te==0))}]"
        )

    # convert accumulated predictions to arrays
    results["y_true_all"] = np.array(results["y_true_all"], dtype=np.int32)
    results["y_pred_all"] = np.array(results["y_pred_all"], dtype=np.int32)
    results["y_prob_all"] = np.array(results["y_prob_all"], dtype=np.float32)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SHAP FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

def compute_and_plot_shap(
    model: xgb.XGBClassifier,
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute SHAP values on a 2000-window sample and save bar chart.
    Returns mean absolute SHAP per feature (length 16).
    """
    print("\n  Computing SHAP values …")
    rng    = np.random.default_rng(0)
    idx    = rng.choice(len(X), size=min(2000, len(X)), replace=False)
    X_samp = X[idx]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_samp)

    # mean |SHAP| per feature → overall importance
    mean_shap = np.abs(shap_values).mean(axis=0)
    order     = np.argsort(mean_shap)[::-1]   # descending

    # ── bar chart ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7))
    palette = ["#c0392b" if r < 5 else "#2980b9" for r in range(N_FEATURES)]
    # map feature index → rank in descending order
    color_map = {feat_idx: palette[rank] for rank, feat_idx in enumerate(order)}

    bars = ax.barh(
        [FEATURE_NAMES[i] for i in order[::-1]],   # bottom→top ascending
        mean_shap[order[::-1]],
        color=[color_map[i] for i in order[::-1]],
        edgecolor="white", linewidth=0.4,
    )
    ax.set_xlabel("Mean |SHAP value|  (average impact on model output)", fontsize=11)
    ax.set_title(
        "XGBoost — Feature Importance (SHAP)\n"
        "Red = top 5 predictors of distress  |  Blue = remaining features",
        fontsize=12,
    )
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    out = DOCS_DIR / "shap_feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP chart saved → {out}")

    # ── print top 10 ─────────────────────────────────────────────────────
    print("\n  Top 10 features by mean |SHAP|:")
    for rank, feat_idx in enumerate(order[:10], 1):
        print(f"    {rank:2d}. {FEATURE_NAMES[feat_idx]:<28}  {mean_shap[feat_idx]:.5f}")

    return mean_shap


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_loso_f1(results: dict) -> None:
    sids    = results["subject_ids"]
    f1s     = results["f1"]
    mean_f1 = float(np.mean(f1s))
    target  = 0.78

    fig, ax = plt.subplots(figsize=(13, 5))
    colors  = ["#27ae60" if f >= target else "#e74c3c" for f in f1s]
    ax.bar([f"S{s}" for s in sids], f1s, color=colors, edgecolor="white", width=0.6)
    ax.axhline(target,   color="darkorange", linestyle="--", linewidth=1.8,
               label=f"Target F1 = {target}")
    ax.axhline(mean_f1,  color="navy",       linestyle="-",  linewidth=1.8,
               label=f"Mean F1   = {mean_f1:.4f}")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 Score (distress class)", fontsize=11)
    ax.set_title(
        "XGBoost LOSO-CV — Per-Subject F1 on Distress Class\n"
        "Green ≥ 0.78 target  |  Red = below target",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = DOCS_DIR / "xgb_loso_cv_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  LOSO-CV plot saved → {out}")


def plot_confusion_matrix(results: dict) -> None:
    y_true = results["y_true_all"]
    y_pred = results["y_pred_all"]
    cm     = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    labels = ["Safe (0)", "Distress (1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title("XGBoost — Confusion Matrix\n(all 15 LOSO folds combined)", fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}",
                    ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    out = DOCS_DIR / "xgb_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {out}")


def plot_roc_curve(results: dict) -> None:
    from sklearn.metrics import roc_curve as sk_roc
    y_true = results["y_true_all"]
    y_prob = results["y_prob_all"]
    auc    = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = sk_roc(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2980b9", linewidth=2, label=f"XGBoost  AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("XGBoost — ROC Curve (all LOSO folds combined)", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = DOCS_DIR / "xgb_roc_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curve saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "=" * 65)
    print("  Phase 1.5 — XGBoost Baseline Classifier")
    print("  GPU: NVIDIA GeForce MX150  |  CUDA 12.4  |  XGBoost 3.x")
    print("=" * 65)

    # ── 1. device ───────────────────────────────────────────────────────
    device = detect_device()

    # ── 2. data ─────────────────────────────────────────────────────────
    print("\n  Loading dataset …")
    X, y, groups = load_dataset()
    n_dist = int(np.sum(y == 1))
    n_safe = int(np.sum(y == 0))

    # ── 3. LOSO-CV ───────────────────────────────────────────────────────
    results = run_loso_cv(X, y, groups, device)

    mean_f1  = float(np.mean(results["f1"]))
    mean_pre = float(np.mean(results["precision"]))
    mean_rec = float(np.mean(results["recall"]))
    mean_auc = float(np.mean(results["roc_auc"]))
    passed   = mean_f1 >= 0.78

    print("\n" + "=" * 65)
    print("  LOSO-CV RESULTS (15 folds)")
    print("=" * 65)
    print(f"  Mean F1        : {mean_f1:.4f}  "
          f"{'✓ PASS — above 0.78 target' if passed else '✗ FAIL — below 0.78 target'}")
    print(f"  Mean Precision : {mean_pre:.4f}")
    print(f"  Mean Recall    : {mean_rec:.4f}")
    print(f"  Mean ROC-AUC   : {mean_auc:.4f}")
    print(f"  Best  F1 fold  : {max(results['f1']):.4f}  "
          f"(S{results['subject_ids'][int(np.argmax(results['f1']))]})")
    print(f"  Worst F1 fold  : {min(results['f1']):.4f}  "
          f"(S{results['subject_ids'][int(np.argmin(results['f1']))]})")

    if not passed:
        print("\n  WARNING: F1 below 0.78. Inspect preprocessing before training LSTM.")

    # ── 4. Final model on all data (for SHAP + deployment) ───────────────
    print("\n  Training final model on ALL subjects for SHAP …")
    spw_all     = n_safe / max(n_dist, 1)
    final_model = build_model(scale_pos_weight=spw_all, device=device)

    # 80/20 split just to satisfy early_stopping_rounds
    n_tr = int(0.8 * len(X))
    rng  = np.random.default_rng(42)
    idx  = rng.permutation(len(X))
    final_model.fit(
        X[idx[:n_tr]], y[idx[:n_tr]],
        eval_set=[(X[idx[n_tr:]], y[idx[n_tr:]])],
        verbose=False,
    )
    print(f"  Best iteration : {final_model.best_iteration}")

    # ── 5. Save model ────────────────────────────────────────────────────
    model_path = MODELS_DIR / "baseline_xgb.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"\n  Model saved → {model_path}")

    # ── 6. SHAP ──────────────────────────────────────────────────────────
    mean_shap = compute_and_plot_shap(final_model, X)

    # ── 7. Plots ─────────────────────────────────────────────────────────
    plot_loso_f1(results)
    plot_confusion_matrix(results)
    plot_roc_curve(results)

    # ── 8. Checklist ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 1.5 CHECKLIST")
    print("=" * 65)
    print(f"  [{'x' if passed else ' '}] XGBoost F1 > 0.78 on distress (LOSO-CV): "
          f"{mean_f1:.4f}")
    print(f"  [x] scale_pos_weight used: {spw_all:.2f}")
    print(f"  [x] SHAP values computed — top feature: {FEATURE_NAMES[int(np.argmax(mean_shap))]}")
    print(f"  [x] Model saved to biometric_ml/models/baseline_xgb.pkl")
    print(f"  [x] 4 plots saved to docs/")
    print()

    if passed:
        print("  Feature engineering VALIDATED.")
        print("  Safe to proceed to Phase 1.6 — LSTM Danger Classifier.")
    print()


if __name__ == "__main__":
    main()
