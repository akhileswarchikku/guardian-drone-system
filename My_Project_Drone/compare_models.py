"""
Quick model comparison: XGBoost (Phase 1.5) vs LSTM (Phase 1.6)
Reads XGBoost results from docs/loso_results.json.
LSTM results are from the recorded training run (bbf3jwq5q).
Run: C:/Users/akhil/anaconda3/envs/LLM_GPU/python.exe compare_models.py
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent

# ── XGBoost results (docs/loso_results.json, most recent calibrated run) ─────
with open(ROOT / "docs" / "loso_results.json") as f:
    xgb_raw = json.load(f)

subjects   = xgb_raw["subject_ids"]          # [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]
xgb_f1     = xgb_raw["f1"]
xgb_prec   = xgb_raw["precision"]
xgb_rec    = xgb_raw["recall"]
xgb_auc    = xgb_raw["roc_auc"]

# ── LSTM results (from training run output, 2026-06-08) ───────────────────────
# Per-subject order matches subjects list above
lstm_f1   = [0.425, 0.025, 0.744, 0.525, 0.059, 0.225, 0.527, 0.575,
             0.049, 0.361, 0.756, 0.553, 0.575, 0.792, 0.386]
lstm_prec = [0.679, 0.057, 0.963, 0.362, 0.128, 0.463, 1.000, 0.644,
             0.222, 0.290, 0.608, 0.859, 0.739, 0.842, 0.645]
lstm_rec  = [0.309, 0.016, 0.606, 0.953, 0.038, 0.148, 0.358, 0.519,
             0.028, 0.478, 1.000, 0.407, 0.471, 0.748, 0.276]
lstm_auc  = [0.948, 0.775, 0.994, 0.958, 0.750, 0.903, 0.953, 0.952,
             0.729, 0.812, 0.989, 0.940, 0.929, 0.979, 0.884]

# ── Summary table ─────────────────────────────────────────────────────────────
labels = [f"S{s}" for s in subjects]

def mean(lst): return sum(lst) / len(lst)

print("\n" + "="*72)
print("  MODEL COMPARISON — XGBoost (Phase 1.5) vs LSTM (Phase 1.6)")
print("="*72)
print(f"\n{'Subject':<8} {'XGB F1':>7} {'LSTM F1':>8} {'dF1':>7} | "
      f"{'XGB AUC':>8} {'LSTM AUC':>9} {'dAUC':>7}")
print("-"*72)

for i, sid in enumerate(subjects):
    df1  = lstm_f1[i]  - xgb_f1[i]
    dauc = lstm_auc[i] - xgb_auc[i]
    win_f1  = "^" if lstm_f1[i]  > xgb_f1[i]  else ("v" if lstm_f1[i]  < xgb_f1[i]  else "=")
    win_auc = "^" if lstm_auc[i] > xgb_auc[i] else ("v" if lstm_auc[i] < xgb_auc[i] else "=")
    print(f"  S{sid:<5} {xgb_f1[i]:>7.3f} {lstm_f1[i]:>8.3f} {df1:>+7.3f}{win_f1} | "
          f"{xgb_auc[i]:>8.3f} {lstm_auc[i]:>9.3f} {dauc:>+7.3f}{win_auc}")

print("-"*72)
mxf1  = mean(xgb_f1);   mlf1  = mean(lstm_f1)
mxp   = mean(xgb_prec); mlp   = mean(lstm_prec)
mxr   = mean(xgb_rec);  mlr   = mean(lstm_rec)
mxa   = mean(xgb_auc);  mla   = mean(lstm_auc)
print(f"  {'MEAN':<6} {mxf1:>7.4f} {mlf1:>8.4f} {mlf1-mxf1:>+7.4f}  | "
      f"{mxa:>8.4f} {mla:>9.4f} {mla-mxa:>+7.4f}")

print(f"\n{'Metric':<18} {'XGBoost':>10} {'LSTM':>10} {'Winner':>10}")
print("-"*52)
rows = [
    ("Mean F1",        mxf1, mlf1),
    ("Mean Precision", mxp,  mlp),
    ("Mean Recall",    mxr,  mlr),
    ("Mean AUC",       mxa,  mla),
    ("Best Fold F1",   max(xgb_f1), max(lstm_f1)),
    ("Worst Fold F1",  min(xgb_f1), min(lstm_f1)),
]
for name, xv, lv in rows:
    if   abs(xv - lv) < 0.001: winner = "Tie"
    elif xv > lv:               winner = "XGBoost"
    else:                       winner = "LSTM"
    print(f"  {name:<16} {xv:>10.4f} {lv:>10.4f} {winner:>10}")

print("\n  ^ = LSTM wins   v = XGBoost wins")
print(f"\n  XGBoost wins F1 on {sum(x>l for x,l in zip(xgb_f1,lstm_f1))}/15 subjects")
print(f"  LSTM    wins F1 on {sum(l>x for x,l in zip(xgb_f1,lstm_f1))}/15 subjects")
print(f"  XGBoost wins AUC on {sum(x>l for x,l in zip(xgb_auc,lstm_auc))}/15 subjects")
print(f"  LSTM    wins AUC on {sum(l>x for x,l in zip(xgb_auc,lstm_auc))}/15 subjects")
print()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(subjects))
w = 0.35

for ax, (xgb_vals, lstm_vals, title, ylim) in zip(axes, [
    (xgb_f1,  lstm_f1,  "F1 Score (Distress Class) — LOSO-CV", (0, 1.05)),
    (xgb_auc, lstm_auc, "ROC-AUC — LOSO-CV",                   (0.5, 1.05)),
]):
    b1 = ax.bar(x - w/2, xgb_vals,  w, label="XGBoost", color="#2196F3", alpha=0.85)
    b2 = ax.bar(x + w/2, lstm_vals, w, label="LSTM",     color="#FF5722", alpha=0.85)
    ax.axhline(np.mean(xgb_vals),  color="#2196F3", ls="--", lw=1.2, alpha=0.6)
    ax.axhline(np.mean(lstm_vals), color="#FF5722", ls="--", lw=1.2, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylim(*ylim); ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Test Subject (LOSO-CV)")

axes[0].set_ylabel("F1 Score")
axes[1].set_ylabel("AUC")

plt.tight_layout()
out = ROOT / "docs" / "model_comparison.png"
plt.savefig(out, dpi=150)
print(f"  Comparison plot saved -> {out}")

# ── Precision / Recall breakdown ──────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
for ax, (xv, lv, title) in zip(axes2, [
    (xgb_prec, lstm_prec, "Precision (Distress Class)"),
    (xgb_rec,  lstm_rec,  "Recall (Distress Class)"),
]):
    ax.bar(x - w/2, xv, w, label="XGBoost", color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, lv, w, label="LSTM",     color="#FF5722", alpha=0.85)
    ax.axhline(np.mean(xv), color="#2196F3", ls="--", lw=1.2, alpha=0.6)
    ax.axhline(np.mean(lv), color="#FF5722", ls="--", lw=1.2, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylim(0, 1.1); ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Test Subject (LOSO-CV)")

plt.tight_layout()
out2 = ROOT / "docs" / "model_comparison_prec_rec.png"
plt.savefig(out2, dpi=150)
print(f"  Precision/Recall plot saved -> {out2}\n")
