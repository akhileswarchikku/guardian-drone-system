# Model Card — Guardian Drone Danger Classifier

**Version:** 1.6  
**Date:** 2026-06-08  
**Phase:** 1.6 (Baseline LSTM) + 1.7 (Contextual Gate) + 1.8 (Export)

---

## Model Summary

| Item | Detail |
|---|---|
| Architecture | 2-layer LSTM, hidden=128, dropout=0.3 + FC head (128→64→2) |
| Input | (batch, 8, 16) — 8 consecutive 30-second feature windows |
| Output | Logits [safe, distress]; danger score = softmax(distress) × 100 |
| Parameters | 215,234 |
| Training data | WESAD dataset — 15 subjects, Empatica E4 wristband |
| Evaluation | Leave-One-Subject-Out Cross-Validation (LOSO-CV, 15 folds) |

---

## Training Dataset

**WESAD (Wearable Stress and Affect Detection)**  
- 15 healthy adult subjects (S2–S17, S12 missing from original dataset)
- Sensors: Empatica E4 wristband — BVP (64 Hz), EDA (4 Hz), ACC (32 Hz)
- Labels: 1=baseline(safe), 2=stress(distress), 3=amusement, 4=meditation
- Binary mapping: 2 → distress (1), all others → safe (0)
- Windows: 30-second windows, 5-second step → 17,292 total windows
- Class balance: distress=1,994 (11.5%), safe=15,298 (88.5%), ratio ~1:7.7
- Feature sequences: 8 consecutive windows per LSTM input = 40 seconds of context
- Normalization: Z-score per feature using training-fold statistics (per LOSO fold)

**16 features per window:**
- HRV (6): rmssd, sdnn, pnn50, lf_hf, mean_rr, std_rr  
- EDA (6): mean, std, phasic_peaks, phasic_amplitude, slope, tonic_mean  
- ACC (4): mean_mag, std_mag, spectral_entropy, breathing_rate

---

## LOSO-CV Performance

All results are on held-out test subjects (never seen during training).

| Metric | Value | Target | Status |
|---|---|---|---|
| Mean F1 (distress) | 0.4385 | > 0.83 | BELOW TARGET |
| Mean Precision | 0.5668 | — | — |
| Mean Recall | 0.4238 | — | — |
| Mean AUC-ROC | 0.8996 | — | — |
| Best fold | S16: F1=0.792 | — | — |
| Worst fold | S3: F1=0.025 | — | — |

**Per-subject breakdown:**

| Subject | F1 | Prec | Recall | AUC |
|---|---|---|---|---|
| S2 | 0.425 | 0.679 | 0.309 | 0.948 |
| S3 | 0.025 | 0.057 | 0.016 | 0.775 |
| S4 | 0.744 | 0.963 | 0.606 | 0.994 |
| S5 | 0.525 | 0.362 | 0.953 | 0.958 |
| S6 | 0.059 | 0.128 | 0.038 | 0.750 |
| S7 | 0.225 | 0.463 | 0.148 | 0.903 |
| S8 | 0.527 | 1.000 | 0.358 | 0.953 |
| S9 | 0.575 | 0.644 | 0.519 | 0.952 |
| S10 | 0.049 | 0.222 | 0.028 | 0.729 |
| S11 | 0.361 | 0.290 | 0.478 | 0.812 |
| S13 | 0.756 | 0.608 | 1.000 | 0.989 |
| S14 | 0.553 | 0.859 | 0.407 | 0.940 |
| S15 | 0.575 | 0.739 | 0.471 | 0.929 |
| S16 | 0.792 | 0.842 | 0.748 | 0.979 |
| S17 | 0.386 | 0.645 | 0.276 | 0.884 |

**Why F1 is below 0.83 target:**  
Cross-subject generalization on wrist-only biometrics is hard. Published WESAD
literature reports 0.45–0.65 F1 for LOSO-CV with wrist sensors only. AUC=0.900
confirms the features are discriminative — the model ranks distress windows above
safe windows 90% of the time. The F1 gap is due to class imbalance (1:7.7) and
high inter-subject variability (S3 and S6 have atypical physiology that does not
match training patterns). The Contextual Gating Layer (Phase 1.7) is the
production mechanism for achieving high real-world precision, not this raw F1.

---

## Contextual Gating Layer (Phase 1.7)

The model is never deployed bare. Every prediction passes through `ContextualGate`
before an SOS is dispatched:

| Gate | Rule | Rationale |
|---|---|---|
| Rolling smoother | Average last 10 readings (50s) | Suppresses single-window spikes |
| Score threshold | Smoothed score >= 70 (or 85 in GPS safe zone) | Primary danger threshold |
| Multi-signal | >= 2 of 3 bio-signals (HRV, GSR, breathing) in distress range | Blocks exercise false alarms |
| Duration | Sustained above threshold for >= 15 seconds | Blocks momentary fear spikes |
| Dismiss window | 10-second user cancel after first trigger | Respects user judgment |

**Spec test results:**
- Exercise scenario (elevated breathing, low GSR, 20s) → BLOCKED ✓
- Genuine distress (all 3 signals elevated, 20s) → FIRES ✓

---

## Model Files

| File | Size | Description |
|---|---|---|
| `biometric_ml/models/best_danger_model.pt` | 843.8 KB | FP32 state dict (full precision) |
| `biometric_ml/models/danger_lstm.onnx` | 845.0 KB | ONNX export (opset 17, dynamic batch) |
| `biometric_ml/models/danger_lstm_int8.pt` | 222.7 KB | INT8 dynamic quantization (3.79x smaller) |

**ONNX validation:** max output difference vs PyTorch = 3.81e-06 (tolerance 0.001) ✓  
**Inference time (CPU, 1000 runs):** FP32 mean=1.17ms, INT8 mean=2.22ms — both pass <50ms ✓

Note: INT8 is slower on x86 CPUs because modern desktop processors have highly
optimized FP32 SIMD units. The INT8 advantage is model size (3.79x) for deployment
on edge hardware (ARM Cortex-A, Jetson Orin Nano, future smartwatch chips) where
memory bandwidth is the bottleneck, not compute.

---

## Known Failure Modes

Understanding where this model fails is as important as knowing where it succeeds.

### 1. Exercise (most common false positive)
**What happens:** Running, cycling, or gym workouts elevate HR, HRV variability,
and breathing rate. The LSTM may score exercise windows in the 40–65 range.  
**Mitigation:** Contextual Gate multi-signal check — exercise does not cause the
same GSR spike as genuine fear. With low EDA, multi-signal agreement fails and
SOS is blocked.  
**Residual risk:** High-intensity interval training + anxiety simultaneously could
trigger. Duration gate (15s) provides a second layer.

### 2. Panic attacks and anxiety disorders
**What happens:** Panic attacks produce physiological signatures identical to
genuine external threat — HRV drops, EDA spikes, hyperventilation. The model
will score these as distress (correctly, by its labels) and the gate will fire.  
**Implication:** Not a model failure — the SOS correctly identifies a medical
emergency. A drone dispatch is appropriate. Document this to set user expectations:
the system responds to physiological distress, not just external threats.

### 3. Medical conditions affecting HRV
**Affected conditions:** Atrial fibrillation, diabetes (autonomic neuropathy),
hypothyroidism, beta-blocker medication.  
**What happens:** These conditions suppress or distort HRV patterns. Subjects
with these conditions were excluded from WESAD. Model behavior is undefined.  
**Mitigation:** Require medical screening before deployment. Do not use with
subjects known to have cardiac arrhythmias without cardiologist sign-off.

### 4. Poor wristband contact
**What happens:** Loose E4/equivalent wristband produces noisy BVP (flat or
high-variance signal). Preprocessing rejects these windows as invalid.  
**Mitigation:** `is_valid_window()` in preprocessing.py rejects flat BVP windows.
A high rejection rate surfaces as a warning in the data pipeline.

### 5. Subjects like S3, S6, S10
**What happens:** These subjects have atypical physiology that does not generalize
from the other 14 subjects. F1 < 0.10 on these subjects.  
**Long-term fix:** Enroll personal baseline (5-minute resting calibration per user)
to personalize thresholds. See `biometric_ml.preprocessing.PersonalBaseline`.

### 6. Apple Watch (no EDA sensor)
**What happens:** EDA (skin conductance) is the #1 most important feature. Apple
Watch has no EDA sensor. All EDA features are imputed with safe-class medians,
which biases the model toward predicting safe.  
**Mitigation:** See `docs/hardware_shopping_list.md`. ESP32 + Grove GSR sensor
(~$15) provides continuous EDA equivalent to Empatica E4.

---

## Data Requirements for Deployment

To use this model reliably in production:

| Requirement | Minimum | Recommended |
|---|---|---|
| BVP/PPG | 64 Hz, wrist | Same |
| EDA/GSR | 4 Hz, wrist | Same |
| ACC | 32 Hz, 3-axis, wrist | Same |
| Window length | 30 seconds | Same |
| Inference window | Every 5 seconds | Same |
| Temporal context | 8 windows (40 seconds) | Same |

The model cannot be used with heart rate only (e.g., Apple Watch without add-on
GSR sensor). Heart rate alone does not distinguish distress from exercise.

---

## Intended Use

This model is part of the Guardian Drone System for women's safety in India.
Intended users: adult women who have consented to biometric monitoring and
understand the system's limitations.

**Not intended for:** clinical diagnosis, medical use, children under 18,
subjects with cardiac conditions, or autonomous deployment without human
oversight in the dispatch loop.

---

## Ethical Considerations

- All WESAD subjects provided informed consent under University of Augsburg IRB
- Model should not be used to infer emotional state without user knowledge
- False negatives (missed real threats) are more dangerous than false positives
  (unnecessary drone dispatch) in this application — precision/recall trade-off
  is set accordingly
- Biometric data is processed on-device where possible; GPS coordinates are
  sent to dispatch only when SOS fires, not continuously
