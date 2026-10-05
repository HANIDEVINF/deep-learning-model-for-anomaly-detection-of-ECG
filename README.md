# AI ECG Arrhythmia Classification (v7) — Hybrid 1D-CNN + Transformer + RR Context

> **Strict Inter-Patient MIT-BIH Evaluation (DS1 / DS2) · 97,045 Parameters · 155.4 KB INT8 TFLite (`0.32 ms/beat`)**

## 1. Executive Summary & Why Protocol Comes First
Many published ECG arrhythmia papers report `98%–99%` accuracy on the MIT-BIH Arrhythmia Database by using **intra-patient** random beat splits—where heartbeats from the same patient appear in both training and test sets. Because each patient has a unique QRS morphology, intra-patient splits measure **patient memorization** rather than clinical generalization.

**Version 7** enforces the strict **de Chazal DS1 / DS2 inter-patient protocol**:
- **DS1 (Training & Cross-Validation):** 50,977 beats across 22 patients
- **DS2 (Held-Out Test Set):** 49,668 beats across 22 **completely unseen patients** (zero patient overlap)
- **Validation Strategy:** 5-fold `StratifiedGroupKFold` grouped strictly by patient ID

---

## 2. Quantitative Results on Unseen DS2 Patients (49,668 Beats)

| Evaluation Configuration | Overall DS2 Accuracy | Macro-F1 (N/S/V/F) | Clinical / Engineering Interpretation |
| :--- | :---: | :---: | :--- |
| **Single Raw Model (Fold 1)** | **87.63%** | **0.5172** | Uncalibrated out-of-patient generalization |
| **5-Fold Patient-Grouped Ensemble** | **90.13%** | **0.5290** | Variance reduction across 5 patient-disjoint folds |
| **OOF Prior-Calibrated (τ = 1.5)** | **94.56%** | **0.4910** | High-specificity operating point (analyzed transparently for class-N dominance) |

### Per-Class Inter-Patient Performance (5-Fold Ensemble)
- **Normal (N):** Precision `95.8%` · Recall `93.4%`
- **Ventricular Ectopic (V — PVC):** Sensitivity `86.4%` · Precision `84.1%`
- **Supraventricular Ectopic (S — PAC):** Explicitly modeled via 8 cleaned RR-interval prematurity & compensatory pause ratios

---

## 3. Dual-Branch Neural Architecture (97,045 Parameters)

```text
Input A: Dual-Lead ECG Beat Window (200 samples × 2 leads @ 360 Hz)
  ├─► 0.5–45 Hz Zero-Phase Butterworth Bandpass Filter + Per-Beat Z-Score
  ├─► Conv1D(32, k=7) + BatchNorm + ReLU + MaxPool(2)
  ├─► Conv1D(64, k=5) + BatchNorm + ReLU + MaxPool(2)
  ├─► Conv1D(64, k=3) + BatchNorm + ReLU
  ├─► 2 × Multi-Head Self-Attention Transformer Blocks (4 heads, d_model=64, FFN=128, Dropout=0.2)
  └─► GlobalAveragePooling1D ──► 64-D Morphology Representation

Input B: 8 Cleaned RR-Interval Temporal Context Features (DS1-Standardized)
  ├─► [prev_RR, next_RR, local_mean_10, prematurity_ratio, pause_ratio, diff_prev, diff_next, median_norm]
  └─► Dense(32, ReLU) ──► Dense(16, ReLU) ──► 16-D Rhythm Context Representation

Fusion & Classification Head:
  └─► Concatenate([64-D Morphology, 16-D Rhythm] = 80-D)
      └─► Dense(64, ReLU) + Dropout(0.3) ──► Softmax (5 AAMI Classes: N, S, V, F, Q)
```

---

## 4. Embedded Edge Deployment & TFLite Quantization Verification

| Artifact Format | File Size | Mean Latency / Beat | Argmax Agreement vs Keras Float32 |
| :--- | :---: | :---: | :---: |
| **Keras Float32 (`.keras`)** | 379.08 KB | ~4.10 ms | 100.00% (Reference) |
| **TFLite Dynamic Range** | 146.00 KB | 0.29 ms | 99.94% |
| **TFLite Full INT8 Quantized** | **155.40 KB** | **0.32 ms** | **99.82%** (Verified across 5,000 DS2 beats) |
