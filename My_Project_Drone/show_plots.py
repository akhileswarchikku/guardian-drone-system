"""
Guardian Drone — Visual Results Gallery

Opens all generated plots grouped by phase in separate figure windows.
Each window has a title, description, and all plots for that phase.

Run:
    C:/Users/akhil/anaconda3/envs/LLM_GPU/python.exe show_plots.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

ROOT     = Path(__file__).parent
DOCS_DIR = ROOT / "docs"

# ── Plot registry — grouped by phase ─────────────────────────────────────────
GALLERY = [
    {
        "window_title": "Phase 1.5 — XGBoost Baseline Classifier",
        "description": (
            "XGBoost trained on 16 HRV/EDA/ACC features with LOSO-CV (15 folds).\n"
            "Top feature: eda_tonic_mean (skin conductance level) — 3x higher in distress.\n"
            "Mean F1=0.455  |  Mean AUC=0.888  |  Best: S4 F1=0.813"
        ),
        "plots": [
            ("shap_feature_importance.png",
             "Feature Importance (Gain)\nEDA dominates — top 3 are EDA/HRV"),
            ("xgb_loso_cv_results.png",
             "Per-Subject F1 Score\nOnly S4 reaches target (0.78)"),
            ("xgb_confusion_matrix.png",
             "Confusion Matrix\nHigh recall (few missed threats)"),
            ("xgb_roc_curve.png",
             "ROC Curve\nAUC=0.831 — solid discrimination"),
        ],
        "layout": (2, 2),
    },
    {
        "window_title": "Phase 1.6 — LSTM Danger Classifier",
        "description": (
            "2-layer LSTM on 8-window feature sequences (40s temporal context).\n"
            "Input: (batch, 8 timesteps, 16 features)  |  Trained 30 epochs, AdamW.\n"
            "Mean F1=0.439  |  Mean AUC=0.900  |  Best: S16 F1=0.792  |  S8 Precision=1.000"
        ),
        "plots": [
            ("lstm_loso_cv_results.png",
             "Per-Subject F1 Score\nS4, S13, S16 near or above 0.75"),
            ("lstm_confusion_matrix.png",
             "Confusion Matrix\nPrecision improved vs XGBoost"),
            ("lstm_roc_curve.png",
             "ROC Curve\nAUC=0.882 — better than XGBoost"),
        ],
        "layout": (1, 3),
    },
    {
        "window_title": "Model Comparison — XGBoost vs LSTM",
        "description": (
            "LSTM wins on all metrics with consistent threshold (0.5).\n"
            "LSTM F1 wins: 9/15 subjects  |  LSTM AUC wins: 11/15 subjects\n"
            "Key advantage: LSTM precision 0.567 vs XGBoost 0.443 — fewer false alarms"
        ),
        "plots": [
            ("model_comparison.png",
             "F1 + AUC side-by-side\nLSTM better on most subjects"),
            ("model_comparison_prec_rec.png",
             "Precision + Recall\nLSTM: higher precision, comparable recall"),
        ],
        "layout": (1, 2),
    },
    {
        "window_title": "Phase 1.7 — Contextual Gate + End-to-End Scenario Test",
        "description": (
            "Full pipeline: LSTM score → 5-gate ContextualGate → SOS decision.\n"
            "Exercise (5 min running): scores 50-56, gate NEVER fires — CORRECT.\n"
            "Genuine distress (attack): EDA spikes to 2.1x safe level, SOS fires at t=220s — CORRECT."
        ),
        "plots": [
            ("scenario_test.png",
             "Exercise vs Distress (5 min each)\nGreen = gate held  |  Red vertical = SOS fired"),
        ],
        "layout": (1, 1),
    },
    {
        "window_title": "Apple Watch — Your Personal 10-Month Stress Timeline",
        "description": (
            "Your Apple Watch export (Jul 2025 – Jun 2026): 56K HR records → 153 prediction windows.\n"
            "All predictions: SAFE — correct, because Apple Watch has no EDA/GSR sensor.\n"
            "EDA is the #1 feature. Fix: ESP32 + Grove GSR sensor (~$15) from hardware_shopping_list.md"
        ),
        "plots": [
            ("apple_watch_timeline.png",
             "10 months of your HR data → danger scores\nAll safe (EDA imputed — no real GSR)"),
        ],
        "layout": (1, 1),
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_image(filename: str):
    path = DOCS_DIR / filename
    if not path.exists():
        return None
    return mpimg.imread(str(path))


def _render_window(entry: dict) -> plt.Figure | None:
    rows, cols = entry["layout"]
    plots      = entry["plots"]

    # Filter to existing files
    available = [(fn, lbl) for fn, lbl in plots if (DOCS_DIR / fn).exists()]
    if not available:
        print(f"  Skipping '{entry['window_title']}' — no files found")
        return None

    fig_h = 4.5 * rows + 1.8   # extra for title + description
    fig   = plt.figure(figsize=(10 * cols, fig_h),
                       num=entry["window_title"])

    # Description banner at top
    fig.text(0.5, 0.97, entry["window_title"],
             ha="center", va="top", fontsize=15, fontweight="bold",
             transform=fig.transFigure)
    fig.text(0.5, 0.93, entry["description"],
             ha="center", va="top", fontsize=9.5,
             color="#444", linespacing=1.6,
             transform=fig.transFigure)

    top_margin = 0.88 if rows == 1 else 0.88

    for idx, (filename, label) in enumerate(available):
        img = _load_image(filename)
        if img is None:
            continue
        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, fontsize=9, pad=6, color="#333")

    plt.subplots_adjust(top=top_margin, hspace=0.08, wspace=0.04,
                        left=0.01, right=0.99, bottom=0.02)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  Guardian Drone — Visual Results Gallery")
    print("=" * 60)
    print(f"  Looking for plots in: {DOCS_DIR}\n")

    existing = sorted(DOCS_DIR.glob("*.png"))
    print(f"  Found {len(existing)} PNG files:")
    for p in existing:
        print(f"    {p.name}")
    print()

    figs = []
    for entry in GALLERY:
        fig = _render_window(entry)
        if fig:
            figs.append(fig)
            print(f"  Window: {entry['window_title']}")

    if not figs:
        print("  No plots found. Run the training scripts first.")
        print("  Order: xgboost_classifier.py -> lstm_classifier.py -> "
              "scenario_test.py")
        return

    print(f"\n  Showing {len(figs)} windows. Close all windows to exit.")
    print("  Tip: maximize each window for full detail.\n")
    plt.show()


if __name__ == "__main__":
    main()
