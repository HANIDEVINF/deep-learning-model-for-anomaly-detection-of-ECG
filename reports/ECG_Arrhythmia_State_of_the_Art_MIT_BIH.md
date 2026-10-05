# State of the Art / État de l'Art — AI ECG Arrhythmia Classification on MIT-BIH

> **Inter-Patient Evaluation, Model Families, Benchmark Comparison & Positioning of Version 7**

## 1. Why the State of the Art Must Start with the Evaluation Protocol
On the MIT-BIH Arrhythmia Database, two studies may report `98%–99%` accuracy while solving materially different evaluation problems. The central issue is **patient separation**:
- **Intra-patient evaluation:** Beats from the same patient appear in both training and test sets, allowing patient-specific QRS morphology to be memorized.
- **Inter-patient protocol (de Chazal DS1 / DS2):** DS1 (22 training recordings, 50,977 beats) and DS2 (22 held-out test recordings, 49,668 beats) contain completely disjoint patients, with paced recordings (`102, 104, 107, 217`) excluded.

> **Reading Rule:** An intra-patient accuracy must never be placed in the same column as an inter-patient accuracy and presented as a direct comparison.

---

## 2. The Five AAMI Classes & DS2 Distribution in v7

| AAMI Class | MIT-BIH Annotation Symbols | Clinical Interpretation | DS1 (Train) | DS2 (Held-Out Test) |
| :---: | :--- | :--- | :---: | :---: |
| **N** | `N, L, R, e, j` | Normal / Bundle branch & escape beats | 45,824 | 44,218 (~89.0%) |
| **S** | `A, a, J, S` | Supraventricular ectopic beats | 943 | 1,836 (~3.7%) |
| **V** | `V, E` | Ventricular ectopic beats (PVC) | 3,788 | 3,219 (~6.5%) |
| **F** | `F` | Fusion of normal and ventricular | 414 | 388 (~0.8%) |
| **Q** | `/, f, Q` | Unknown / Paced | 8 | 7 (~0.01%) |
| **Total** | | | **50,977** | **49,668** |

---

## 3. Main Methodological Comparison Table (MIT-BIH Inter-Patient Literature vs. v7)

| Method / Study | Year | Protocol / Unit | Overall Accuracy | Secondary / Class Metrics | Representation & Architectural Notes |
| :--- | :---: | :--- | :---: | :--- | :--- |
| **de Chazal et al.** | 2004 | DS1/DS2 Inter-patient | 85.90% | S Se: 75.9%, PPV: 38.5%<br/>V Se: 77.7%, PPV: 81.9% | Explicit ECG morphology + RR intervals, linear discriminant classifier |
| **Kachuee et al. (Original)** | 2018 | Original split (not strict DS1/DS2) | 93.40% | — | Deep 1D Residual CNN; transferable representation |
| **Kachuee (Strict Re-run)** | 2018 / Re-eval | Strict DS1/DS2 Inter-patient | 81.20% | S Recall: 0.0%<br/>F Recall: 1.3% | Demonstrates the >12% drop when imposing strict inter-patient separation |
| **Garcia et al.** | 2017 | DS1/DS2 Inter-patient | 92.40% | S PPV: ~53%<br/>V Se: 87.3% | Temporal Vectorcardiogram (TVCG) + Particle Swarm Optimization (PSO) |
| **Li et al. (Residual CNN)** | 2022 | Inter-patient (5-sec segments) | 88.99% | S Se: 35.22%<br/>V Se: 88.35% | Discrete Wavelet Transform (DWT) + Residual CNN + Focal Loss |
| **Wang et al. (CWT + CNN)** | 2021 | Inter-patient | 98.74% | Macro-F1: 68.76%<br/>Mean Se: 67.47% | 2D Continuous Wavelet Transform (CWT) scalogram + 2D CNN + 4 RR features |
| **Zhou et al. (FCBA + RR)** | 2024 | Inter-patient | 95.60% | N/S/V Se: 96.9%, 89.3%, 93.3% | Multiscale convolution + Frequency Convolutional Block Attention + RR |
| **DPFNet (Ma et al.)** | 2026 | Inter-patient MIT-BIH (PVC task) | 98.90% | F1: 92.38%<br/>Spec: 98.97% | Dual-path 1D waveform + 2D wavelet scalogram fusion (PVC-focused) |
| **Project v7 — Single Raw** | **2026** | **Strict DS1/DS2 Inter-patient** | **87.63%** | **Macro-F1 (N/S/V/F): 0.5172**<br/>V Se: 91.3%, N Se: 91.1% | **1D-CNN + 2 Transformer Blocks + 8 Cleaned RR; 97,045 params (155.4 KB INT8)** |
| **Project v7 — 5-Fold Ensemble** | **2026** | **Strict DS1/DS2 Inter-patient** | **90.13%** | **Macro-F1 (N/S/V/F): 0.5290**<br/>V Se: 95.0%, S PPV: 47.2% | **Average of 5 patient-grouped fold models (zero DS2 leakage)** |
| **Project v7 — Calibrated (τ=1.5)** | **2026** | **Strict DS1/DS2 Inter-patient** | **94.56%** | **Macro-F1 (N/S/V/F): 0.4773**<br/>N Se: 99.8%, V PPV: 96.0% | **DS1 OOF prior-adjusted operating point (favors dominant N class)** |

---

## 4. References
1. PhysioNet, *MIT-BIH Arrhythmia Database*, 1.0.0. https://www.physionet.org/content/mitdb/1.0.0/
2. P. de Chazal, M. O'Dwyer, R. B. Reilly, "Automatic classification of heartbeats using ECG morphology and heartbeat interval features," *IEEE Transactions on Biomedical Engineering*, 51(7), 1196–1206, 2004.
3. M. Kachuee, S. Fazeli, M. Sarrafzadeh, "ECG Heartbeat Classification: A Deep Transferable Representation," *IEEE ICHI*, 2018.
4. G. Garcia et al., "Inter-Patient ECG Heartbeat Classification with Temporal VCG Optimized by PSO," *Scientific Reports*, 7, 10543, 2017.
5. Y. Li, R. Qian, K. Li, "Inter-patient arrhythmia classification with improved deep residual convolutional neural network," *Computer Methods and Programs in Biomedicine*, 214, 106582, 2022.
6. T. Wang et al., "Automatic ECG Classification Using Continuous Wavelet Transform and Convolutional Neural Network," *Entropy*, 23(1), 119, 2021.
7. F. Zhou, Y. Sun, Y. Wang, "Inter-patient ECG arrhythmia heartbeat classification network based on multiscale convolution and FCBA," *Biomedical Signal Processing and Control*, 90, 105789, 2024.
8. G. Ma et al., "DPFNet: A dual-path fusion network with attention mechanism for robust premature ventricular contraction detection," *Signal, Image and Video Processing*, 20(3), 150, 2026.
9. Y. Tao, Y. Zhang, "MSAForm: a multiscale-attention transformer with hierarchical feature enhancement for inter-patient arrhythmia detection," *Expert Systems with Applications*, 308, 131157, 2026.
