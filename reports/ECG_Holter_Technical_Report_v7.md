# Technical Report (v7) — AI-Based ECG Holter Analysis and Embedded Arrhythmia Detection

> **Model Architecture, v1→v7 Experimental Evolution, DS1 Out-of-Fold Calibration & TFLite Edge Deployment**

## 1. Executive Summary & v6 → v7 Root-Cause Diagnosis
Version 7 resolves four methodological and architectural bottlenecks identified in v6 (`77.89%` raw accuracy):
1. **Prior-Shift Correction Without Test Leakage:** Training mini-batches are balanced (`N = 34%, S = 28%, V = 19%, F = 17%, Q = 2%`), whereas real Holter recordings are `~89% N`. Version 7 applies Menon et al. (ICLR 2021) post-hoc logit adjustment using **DS1 priors** and selects $\tau$ exclusively on **DS1 out-of-fold (OOF) predictions** ($\tau_{\text{acc}} = 1.5$, $\tau_{F1} = 0.4$).
2. **Elimination of Noisy Early Stopping:** Patient-grouped validation folds contain only 4–6 patients, causing early stopping in v6 to halt at epoch 1–7. Version 7 uses a fixed **40-epoch Cosine Decay schedule** (`initial_lr = 1e-3, alpha = 0.02, clipnorm = 1.0`).
3. **Dual-Lead Selection by Signal Name & Beat-Only RR Cleaning:** Selects `MLII` plus the record's second lead by `sig_name` (`200 × 2` tensor) and filters out non-beat markers (`+, ~, |`) before computing 8 DS1-standardized RR features.

---

## 2. Layer-by-Layer Verified Architecture (`ECG_CNN_Attention_v7`)

| Layer | Output Shape | Parameters | Functional Role |
| :--- | :---: | :---: | :--- |
| `beat_input` (InputLayer) | `(None, 200, 2)` | 0 | Dual-lead ECG window (0.556 s @ 360 Hz), per-lead Z-scored |
| `Conv1D(32, k=7)` + `BN` + `ReLU` + `MaxPool(2)` | `(None, 100, 32)` | 608 | Local waveform & QRS onset feature extraction |
| `Conv1D(64, k=5)` + `BN` + `ReLU` + `MaxPool(2)` | `(None, 50, 64)` | 10,560 | Intermediate morphological motifs |
| `Conv1D(64, k=3)` + `BN` + `ReLU` | `(None, 50, 64)` | 12,608 | Fine-grained 50-position sequence encoding |
| `Transformer Block 1` (4 heads, `key_dim=16`, FFN 128) | `(None, 50, 64)` | 33,472 | Non-local intra-beat self-attention + residual LayerNorm |
| `Transformer Block 2` (4 heads, `key_dim=16`, FFN 128) | `(None, 50, 64)` | 33,472 | Contextual refinement across P-QRS-T wave segments |
| `GlobalAveragePooling1D` + `Dropout(0.2)` | `(None, 64)` | 0 | Compact 64-D morphological embedding |
| `rr_input` → `Dense(32, ReLU)` → `Dense(16, ReLU)` | `(None, 16)` | 816 | 8 DS1-standardized RR interval & rhythm ratios |
| `Concatenate([64, 16])` | `(None, 80)` | 0 | Morphology + temporal rhythm fusion |
| `Dense(64, ReLU)` + `Dropout(0.3)` → `Dense(5, Softmax)` | `(None, 5)` | 5,509 | 5-class AAMI probability output (N, S, V, F, Q) |
| **Total Parameters** | | **97,045** | **96,725 trainable · 320 non-trainable (379.08 KB float32)** |

---

## 3. Complete DS2 Inter-Patient Evaluation Table (49,668 Unseen Beats)

| Configuration | DS2 Accuracy | Macro-F1 (N/S/V/F) | Se (N) | Se (S) | Se (V) | Se (F) | PPV (N) | PPV (S) | PPV (V) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single Model · Raw ($\tau = 0$)** | **87.63%** | **0.5172** | 91.1% | 17.1% | 91.3% | 1.0% | 95.2% | 28.4% | 93.1% |
| **Single Model · $\tau_{F1} = 0.4$** | **89.85%** | **0.5074** | 93.8% | 10.8% | 91.1% | 0.8% | 94.9% | 28.6% | 94.4% |
| **Single Model · $\tau_{\text{acc}} = 1.5$** | **94.56%** | **0.4773** | 99.8% | 1.0% | 88.0% | 0.0% | 94.5% | 52.8% | 96.0% |
| **5-Fold Ensemble · Raw ($\tau = 0$)** | **90.13%** | **0.5290** | 93.6% | 16.2% | 95.0% | 1.8% | 95.4% | 47.2% | 90.1% |
| **5-Fold Ensemble · $\tau_{F1} = 0.4$** | **94.40%** | **0.4928** | 99.1% | 2.9% | 93.6% | 0.0% | 94.9% | 43.5% | 95.7% |
| **5-Fold Ensemble · $\tau_{\text{acc}} = 1.5$** | **94.58%** | **0.4727** | 100.0% | 0.0% | 85.9% | 0.0% | 94.3% | 0.0% | 99.1% |

---

## 4. Version History (v1 → v7)

| Version | DS2 Accuracy | Macro-F1 | Primary Experimental Diagnosis |
| :---: | :---: | :---: | :--- |
| **v1** | 75.30% | 0.370 | Unshuffled validation tail; extreme class-Q weight destabilized gradients |
| **v2** | 88.70% | 0.350 | Stratified by class rather than patient ID (intra-patient validation leakage) |
| **v3** | 77.30% | 0.340 | 3-patient validation split too noisy for model selection |
| **v4** | 84.85% | 0.360 | N/V-dominated mini-batches and weak focal $\gamma$ |
| **v5** | 81.25% | 0.360 | Cross-validated epoch selection remained unstable across patient folds |
| **v6** | 77.89% raw / 92.91% ($\tau=1$) | 0.377 | Severe prior shift; early stopping collapsed to 7 epochs; $\tau$ tuned on DS2 |
| **v7** | **87.63% raw / 90.13% ens / 94.56% ($\tau=1.5$)** | **0.5172 / 0.5290** | **Dual-lead by name, beat-only 8 RR features, fixed 40-epoch cosine schedule, DS1-only OOF calibration, TFLite verification** |

---

## 5. Embedded TensorFlow Lite Deployment & Quantization Fidelity

- **Keras Float32 Artifact:** `379.08 KB` (`97,045` parameters)
- **Dynamic-Range TFLite Artifact:** `146.0 KB`
- **Full-Integer INT8 TFLite Artifact:** `155.4 KB`
- **Measured Inference Latency:** `0.32 ms / beat` (Colab CPU)
- **Quantization Fidelity Check (5,000 DS2 Beats):** `99.82%` argmax decision agreement between Keras Float32 (`88.64%` subset accuracy) and TFLite (`88.72%` subset accuracy).
