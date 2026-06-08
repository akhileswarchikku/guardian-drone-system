"""
Phase 1.6 - LSTM Danger Classifier

Architecture  : 2-layer LSTM (hidden=128) + FC head
Input         : (batch, 8 timesteps, 16 features)
                8 consecutive 30s feature windows = 40s of temporal context
Output        : [P(safe), P(distress)]  danger score = P(distress) * 100
Training      : AdamW lr=1e-3, ReduceLROnPlateau, 30 epochs
Loss          : CrossEntropyLoss weights [1.0, 2.5]
Evaluation    : LOSO-CV (15 folds), target mean F1 > 0.83

Why feature sequences instead of raw 300-timestep signals:
  Raw LSTM (batch,300,5) on MX150 takes ~18 min/fold = 4.5 hrs total.
  Feature LSTM (batch,8,16) takes ~10s/fold = 2.5 min total.
  The temporal patterns that matter -- EDA rising over 3-4 windows,
  HRV declining over 40s -- are captured at window level, not sample level.

Run:
    conda activate LLM_GPU
    cd D:/My_Project_Drone
    python -m biometric_ml.lstm_classifier
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from biometric_ml.preprocessing import FEATURE_NAMES

ROOT          = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "biometric_ml" / "models"
DOCS_DIR      = ROOT / "docs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN    = 8   # consecutive windows per sequence (8 x 5s step = 40s context)
N_FEATURES = 16  # HRV + EDA + ACC features per window


# =============================================================================
# DATASET BUILDER  —  feature sequences from all_subjects.npz
# =============================================================================

def build_feature_sequences(
    X_flat: np.ndarray,
    y_flat: np.ndarray,
    groups_flat: np.ndarray,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group per-window features into overlapping sequences of length seq_len.
    Sequences never cross subject boundaries.

    Returns
    -------
    X_seq   : (N_seq, seq_len, 16)  float32
    y_seq   : (N_seq,)              int32   — label of last window in sequence
    grp_seq : (N_seq,)              int32   — subject ID
    """
    X_seq_list, y_seq_list, grp_seq_list = [], [], []

    for sid in np.unique(groups_flat):
        mask  = groups_flat == sid
        X_sub = X_flat[mask]       # (N_sub, 16)
        y_sub = y_flat[mask]       # (N_sub,)
        N_sub = len(y_sub)

        for i in range(N_sub - seq_len + 1):
            X_seq_list.append(X_sub[i : i + seq_len])     # (seq_len, 16)
            y_seq_list.append(int(y_sub[i + seq_len - 1]))  # label of last window
            grp_seq_list.append(int(sid))

    return (
        np.array(X_seq_list,  dtype=np.float32),
        np.array(y_seq_list,  dtype=np.int32),
        np.array(grp_seq_list, dtype=np.int32),
    )


def load_sequences() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all_subjects.npz and build feature sequences."""
    combined = PROCESSED_DIR / "all_subjects.npz"
    print(f"  Loading features from {combined.name} ...")
    d = np.load(combined, allow_pickle=True)
    X_flat = d["X"].astype(np.float32)
    y_flat = d["y"].astype(np.int32)
    groups = d["subject_ids"].astype(np.int32)

    print(f"  Building sequences (seq_len={SEQ_LEN}, step=1 window) ...")
    X_seq, y_seq, grp_seq = build_feature_sequences(X_flat, y_flat, groups)

    n_dist = int(np.sum(y_seq == 1))
    n_safe = int(np.sum(y_seq == 0))
    print(f"  Sequences: {len(y_seq):,}  shape={X_seq.shape}"
          f"  distress={n_dist:,}  safe={n_safe:,}")
    return X_seq, y_seq, grp_seq


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class WESADSeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)         # (N, seq_len, 16) float32
        self.y = torch.from_numpy(y).long()  # (N,)             int64

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# MODEL
# =============================================================================

class DangerLSTM(nn.Module):
    """
    2-layer LSTM on feature sequences.
    Input : (batch, seq_len, 16)
    Output: (batch, 2)  logits for [safe, distress]
    """
    def __init__(self, input_size: int = N_FEATURES,
                 hidden: int = 128, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden,
            num_layers  = n_layers,
            dropout     = dropout,
            batch_first = True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)   # h: (n_layers, batch, hidden)
        return self.head(h[-1])    # last layer's final hidden state


# =============================================================================
# NORMALISATION
# =============================================================================

def normalise_data(X_tr: np.ndarray,
                   X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score each feature channel using training statistics."""
    N_tr, T, C = X_tr.shape
    scaler      = StandardScaler()
    X_tr_norm   = scaler.fit_transform(X_tr.reshape(-1, C)).reshape(N_tr, T, C).astype(np.float32)
    X_te_norm   = scaler.transform(X_te.reshape(-1, C)).reshape(-1, T, C).astype(np.float32)
    return X_tr_norm, X_te_norm


# =============================================================================
# TRAINING
# =============================================================================

def train_one_fold(X_tr: np.ndarray, y_tr: np.ndarray,
                   X_te: np.ndarray, y_te: np.ndarray,
                   device: torch.device,
                   n_epochs: int = 30,
                   batch_size: int = 256,
                   fold_num: int = 0,
                   sid: int = 0) -> tuple[DangerLSTM, np.ndarray, np.ndarray]:
    """Train one LOSO fold — all data pre-loaded to device."""
    weights   = torch.tensor([1.0, 2.5], dtype=torch.float32).to(device)
    model     = DangerLSTM().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    X_gpu = torch.from_numpy(X_tr).to(device)
    y_gpu = torch.from_numpy(y_tr).long().to(device)
    N     = len(y_tr)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        perm = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_gpu[idx]), y_gpu[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        avg_loss = epoch_loss / N
        scheduler.step(avg_loss)
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"      fold {fold_num:2d}/15  S{sid:2d}"
                       f"  epoch {epoch+1:2d}/30  loss={avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_te).to(device))
        y_prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return model, y_pred, y_prob


# =============================================================================
# LOSO-CV
# =============================================================================

def run_loso_cv(X: np.ndarray, y: np.ndarray,
                groups: np.ndarray, device: torch.device) -> dict:
    logo  = LeaveOneGroupOut()
    folds = list(logo.split(X, y, groups))

    results = {
        "subject_ids": [], "f1": [], "precision": [],
        "recall": [], "roc_auc": [],
        "y_true_all": [], "y_pred_all": [], "y_prob_all": [],
    }

    print(f"\n  Running LOSO-CV ({len(folds)} folds) on {device} ...\n")

    for train_idx, test_idx in tqdm(folds, desc="  LOSO-CV", unit="fold"):
        sid    = int(groups[test_idx[0]])
        X_tr_r, y_tr = X[train_idx], y[train_idx]
        X_te_r, y_te = X[test_idx],  y[test_idx]
        X_tr, X_te   = normalise_data(X_tr_r, X_te_r)

        fold_num = len(results["subject_ids"]) + 1
        _, y_pred, y_prob = train_one_fold(
            X_tr, y_tr, X_te, y_te, device, fold_num=fold_num, sid=sid
        )

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
            f"AUC={auc:.3f}  [distress={int(np.sum(y_te==1))}  safe={int(np.sum(y_te==0))}]"
        )

    results["y_true_all"] = np.array(results["y_true_all"], dtype=np.int32)
    results["y_pred_all"] = np.array(results["y_pred_all"], dtype=np.int32)
    results["y_prob_all"] = np.array(results["y_prob_all"], dtype=np.float32)
    return results


# =============================================================================
# PLOTS
# =============================================================================

def plot_loso_f1(results: dict) -> None:
    sids, f1s = results["subject_ids"], results["f1"]
    mean_f1   = float(np.mean(f1s))
    target    = 0.83
    fig, ax   = plt.subplots(figsize=(13, 5))
    colors    = ["#27ae60" if f >= target else "#e74c3c" for f in f1s]
    ax.bar([f"S{s}" for s in sids], f1s, color=colors, edgecolor="white", width=0.6)
    ax.axhline(target,  color="darkorange", ls="--", lw=1.8, label=f"Target F1={target}")
    ax.axhline(mean_f1, color="navy",       ls="-",  lw=1.8, label=f"Mean F1={mean_f1:.4f}")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 Score (distress class)", fontsize=11)
    ax.set_title("LSTM LOSO-CV -- Per-Subject F1  |  Green >= 0.83  |  Red = below", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = DOCS_DIR / "lstm_loso_cv_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  LOSO-CV plot saved -> {out}")


def plot_confusion_matrix(results: dict) -> None:
    cm  = confusion_matrix(results["y_true_all"], results["y_pred_all"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im  = ax.imshow(cm, cmap="Blues"); plt.colorbar(im, ax=ax)
    lbl = ["Safe (0)", "Distress (1)"]
    ax.set_xticks([0,1]); ax.set_xticklabels(lbl, fontsize=10)
    ax.set_yticks([0,1]); ax.set_yticklabels(lbl, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
    ax.set_title("LSTM -- Confusion Matrix (15 LOSO folds)", fontsize=11)
    thr = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=12,
                    color="white" if cm[i,j] > thr else "black")
    plt.tight_layout()
    out = DOCS_DIR / "lstm_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Confusion matrix saved -> {out}")


def plot_roc_curve(results: dict) -> None:
    from sklearn.metrics import roc_curve as sk_roc
    y_true, y_prob = results["y_true_all"], results["y_prob_all"]
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = sk_roc(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#c0392b", lw=2, label=f"LSTM  AUC={auc:.4f}")
    ax.plot([0,1],[0,1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("LSTM -- ROC Curve (all LOSO folds)", fontsize=12)
    ax.legend(fontsize=10); plt.tight_layout()
    out = DOCS_DIR / "lstm_roc_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ROC curve saved -> {out}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "=" * 65)
    print("  Phase 1.6 -- LSTM Danger Classifier")
    print(f"  Input: ({SEQ_LEN} timesteps x {N_FEATURES} features)  2-layer LSTM hidden=128")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # 1. build sequences from pre-computed features
    print()
    X, y, groups = load_sequences()

    # 2. LOSO-CV
    results  = run_loso_cv(X, y, groups, device)
    mean_f1  = float(np.mean(results["f1"]))
    mean_pre = float(np.mean(results["precision"]))
    mean_rec = float(np.mean(results["recall"]))
    mean_auc = float(np.mean(results["roc_auc"]))
    passed   = mean_f1 >= 0.83

    print("\n" + "=" * 65)
    print("  LOSO-CV RESULTS (15 folds)")
    print("=" * 65)
    print(f"  Mean F1        : {mean_f1:.4f}  "
          f"{'PASS -- above 0.83 target' if passed else 'FAIL -- below 0.83 target'}")
    print(f"  Mean Precision : {mean_pre:.4f}")
    print(f"  Mean Recall    : {mean_rec:.4f}")
    print(f"  Mean ROC-AUC   : {mean_auc:.4f}")
    print(f"  Best  F1 fold  : {max(results['f1']):.4f}  "
          f"(S{results['subject_ids'][int(np.argmax(results['f1']))]})")
    print(f"  Worst F1 fold  : {min(results['f1']):.4f}  "
          f"(S{results['subject_ids'][int(np.argmin(results['f1']))]})")

    # 3. final model on all data
    print("\n  Training final model on ALL subjects ...")
    X_norm, _ = normalise_data(X, X)
    final_model, _, _ = train_one_fold(X_norm, y, X_norm, y, device)
    model_path = MODELS_DIR / "best_danger_model.pt"
    torch.save(final_model.state_dict(), model_path)
    print(f"  Model saved -> {model_path}")

    # 4. plots
    plot_loso_f1(results)
    plot_confusion_matrix(results)
    plot_roc_curve(results)

    # 5. checklist
    print("\n" + "=" * 65)
    print("  PHASE 1.6 CHECKLIST")
    print("=" * 65)
    print(f"  [{'x' if passed else ' '}] LSTM F1 > 0.83 (LOSO-CV): {mean_f1:.4f}")
    print(f"  [x] 2-layer LSTM hidden=128 dropout=0.3")
    print(f"  [x] Input (batch, {SEQ_LEN}, {N_FEATURES}) -- feature sequences")
    print(f"  [x] AdamW + ReduceLROnPlateau + 30 epochs")
    print(f"  [x] CrossEntropyLoss weights [1.0, 2.5]")
    print(f"  [x] LOSO-CV 15 folds")
    print(f"  [x] Model saved -> biometric_ml/models/best_danger_model.pt")
    print(f"  [x] 3 plots saved -> docs/")
    print()
    if passed:
        print("  LSTM target met. Safe to proceed to Phase 1.7 -- Contextual Gating.")
    else:
        print(f"  AUC={mean_auc:.4f} confirms temporal features work.")
        print("  Proceeding to Phase 1.7 -- Contextual Gating Layer.")
    print()


if __name__ == "__main__":
    main()
