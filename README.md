# AI ECG Arrhythmia Classification (v7) — Hybrid 1D-CNN + Transformer + 8 RR Context

> **Strict Inter-Patient MIT-BIH Evaluation (de Chazal DS1 / DS2) · 97,045 Parameters · 155.4 KB INT8 TFLite (`0.32 ms/beat`, `99.82%` Argmax Fidelity)**

## 📂 Repository Structure

- **[`scripts/train_ecg_arrhythmia_v7.py`](./scripts/train_ecg_arrhythmia_v7.py)** — Complete end-to-end Python pipeline: PhysioNet MIT-BIH download, 0.5–45 Hz zero-phase Butterworth filtering, dual-lead selection by `sig_name` (`MLII` + second lead), beat-only 8 RR feature extraction, 5-fold `StratifiedGroupKFold` by patient ID, DS1-only out-of-fold (OOF) logit adjustment, and TFLite INT8 quantization + fidelity verification.
- **[`notebooks/ECG_Arrhythmia_Classification_v7.ipynb`](./notebooks/ECG_Arrhythmia_Classification_v7.ipynb)** — Executed v7 Colab notebook with full DS1/DS2 benchmark outputs.
- **[`reports/ECG_Holter_Technical_Report_v7.md`](./reports/ECG_Holter_Technical_Report_v7.md)** — Full v7 Technical Report (layer-by-layer verified architecture, v1→v7 ablation history, confusion matrix analysis, and embedded TFLite verification).
- **[`reports/ECG_Arrhythmia_State_of_the_Art_MIT_BIH.md`](./reports/ECG_Arrhythmia_State_of_the_Art_MIT_BIH.md)** — State-of-the-Art (SOTA) comparative literature review (2004–2026) and methodological comparison table.
- **[`deployment_constants.json`](./deployment_constants.json)** — Exported DS1 RR standardization parameters (`mean / std`), class priors, and $\tau$ logit-bias vectors for Raspberry Pi / embedded Holter inference.

---

## 1. Executive Summary & Why Protocol Comes First
Many published ECG arrhythmia papers report `98%–99%` accuracy on the MIT-BIH Arrhythmia Database by using **intra-patient** random beat splits—where heartbeats from the same patient appear in both training and test sets. Because each patient has a unique QRS morphology, intra-patient splits measure **patient memorization** rather than clinical generalization.

**Version 7** enforces the strict **de Chazal DS1 / DS2 inter-patient protocol**:
- **DS1 (Training & Cross-Validation):** 50,977 beats across 22 patients
- **DS2 (Held-Out Test Set):** 49,668 beats across 22 **completely unseen patients** (zero patient overlap; paced records `102, 104, 107, 217` excluded)
- **Validation Strategy:** 5-fold `StratifiedGroupKFold` grouped strictly by patient ID

---

## 2. Quantitative Results on Unseen DS2 Patients (49,668 Beats)

| Evaluation Configuration | Overall DS2 Accuracy | Macro-F1 (N/S/V/F) | Se (N) | Se (S) | Se (V) | PPV (N) | PPV (S) | PPV (V) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single Raw Model ($\tau = 0$)** | **87.63%** | **0.5172** | 91.1% | 17.1% | 91.3% | 95.2% | 28.4% | 93.1% |
| **Single Model ($\tau_{F1} = 0.4$)** | **89.85%** | **0.5074** | 93.8% | 10.8% | 91.1% | 94.9% | 28.6% | 94.4% |
| **5-Fold Patient-Grouped Ensemble** | **90.13%** | **0.5290** | 93.6% | 16.2% | 95.0% | 95.4% | 47.2% | 90.1% |
| **Single OOF Prior-Calibrated ($\tau_{\text{acc}} = 1.5$)** | **94.56%** | **0.4773** | 99.8% | 1.0% | 88.0% | 94.5% | 52.8% | 96.0% |

---

## 3. Dual-Branch Neural Architecture (`ECG_CNN_Attention_v7` · 97,045 Params)

```text
Input A: Dual-Lead ECG Beat Window (200 samples × 2 leads @ 360 Hz)
  ├─► 0.5–45 Hz Zero-Phase Butterworth Bandpass Filter + Per-Beat Z-Score
  ├─► Conv1D(32, k=7) + BatchNorm + ReLU + MaxPool(2)   ──► (100, 32)
  ├─► Conv1D(64, k=5) + BatchNorm + ReLU + MaxPool(2)   ──► (50, 64)
  ├─► Conv1D(64, k=3) + BatchNorm + ReLU                ──► (50, 64)
  ├─► 2 × Multi-Head Self-Attention Transformer Blocks (4 heads, key_dim=16, FFN=128, Dropout=0.2)
  └─► GlobalAveragePooling1D + Dropout(0.2)             ──► 64-D Morphology Representation

Input B: 8 Cleaned RR-Interval Temporal Context Features (DS1-Standardized)
  ├─► [prev_rr, next_rr, local_rr, prematurity, compensatory, prev/rec_median, next/rec_median, prev2/local]
  └─► Dense(32, ReLU) ──► Dense(16, ReLU)               ──► 16-D Rhythm Context Representation

Fusion & Classification Head:
  └─► Concatenate([64-D Morphology, 16-D Rhythm] = 80-D)
      └─► Dense(64, ReLU) + Dropout(0.3) ──► Softmax (5 AAMI Classes: N, S, V, F, Q)
```

---

## 4. Embedded Edge Deployment & TFLite Quantization Verification

| Artifact Format | File Size | Mean Latency / Beat | Argmax Agreement vs Keras Float32 |
| :--- | :---: | :---: | :---: |
| **Keras Float32 (`.keras`)** | 379.08 KB | ~4.10 ms | 100.00% (Reference · 88.64% subset acc) |
| **TFLite Dynamic Range** | 146.00 KB | **0.32 ms** | **99.82%** (Verified on 5,000 DS2 beats · 88.72% subset acc) |
| **TFLite Full INT8 Quantized** | **155.40 KB** | **0.32 ms** | **99.82%** |
