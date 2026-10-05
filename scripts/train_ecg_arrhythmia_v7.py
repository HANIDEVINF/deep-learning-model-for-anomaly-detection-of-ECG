#!/usr/bin/env python3
"""
AI-Based ECG Arrhythmia Classification — End-to-End Pipeline (v7)
Strict Inter-Patient MIT-BIH Evaluation (de Chazal DS1 / DS2)
Architecture: 1D-CNN + 2x Multi-Head Transformer + 8 Cleaned RR Features (97,045 params)
"""

import os
import time
import json
import collections
import numpy as np
import wfdb
import scipy.signal as sg
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "mitdb"
FS = 360
PRE = 90
POST = 110
BEAT_LEN = PRE + POST

MERGE_Q_INTO_F = False
USE_TWO_LEADS = True
N_LEADS = 2 if USE_TWO_LEADS else 1
N_RR = 8
TRAIN_EPOCHS = 40
BATCH_SIZE = 256

AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
if MERGE_Q_INTO_F:
    AAMI_MAP = {k: ("F" if v == "Q" else v) for k, v in AAMI_MAP.items()}
    CLASSES = ["N", "S", "V", "F"]
else:
    CLASSES = ["N", "S", "V", "F", "Q"]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
N_CLASSES = len(CLASSES)

# Standard inter-patient split (de Chazal et al., 2004); excludes paced records 102, 104, 107, 217
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

BEAT_SYMBOLS = set(AAMI_MAP)
RR_FEATURE_NAMES = [
    "prev_rr",
    "next_rr",
    "local_rr",
    "prematurity",
    "compensatory",
    "prev_rr/rec_median",
    "next_rr/rec_median",
    "prev2_rr/local",
]


def download_mitdb(dl_dir=DATA_DIR):
    os.makedirs(dl_dir, exist_ok=True)
    if os.path.exists(os.path.join(dl_dir, "100.dat")):
        print("MIT-BIH already present locally, skipping download.")
        return
    print("Downloading MIT-BIH Arrhythmia Database from PhysioNet...")
    wfdb.dl_database("mitdb", dl_dir=dl_dir)
    print("Download complete.")


def bandpass_filter(sig, fs=FS, low=0.5, high=45.0, order=3):
    nyq = 0.5 * fs
    b, a = sg.butter(order, [low / nyq, high / nyq], btype="band")
    return sg.filtfilt(b, a, sig, axis=0)


def select_leads(record):
    """MLII first (selected by NAME, not channel position), then the record's second lead."""
    names = list(record.sig_name)
    i0 = names.index("MLII") if "MLII" in names else 0
    chans = [i0]
    if USE_TWO_LEADS:
        chans += [i for i in range(len(names)) if i != i0][:1]
    return record.p_signal[:, chans]


def extract_beats_from_record(record_id, dl_dir=DATA_DIR):
    rec_path = os.path.join(dl_dir, str(record_id))
    record = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "atr")
    sig = bandpass_filter(select_leads(record))

    keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol], dtype=bool)
    r_samples = ann.sample[keep]
    symbols = [s for s, k in zip(ann.symbol, keep) if k]
    n = len(r_samples)
    rec_median_rr = float(np.median(np.diff(r_samples)) / FS)
    eps = 1e-6

    beats, labels, rr_feats, groups = [], [], [], []
    for i in range(1, n - 1):
        center = r_samples[i]
        start, end = center - PRE, center + POST
        if start < 0 or end > len(sig):
            continue
        beat = sig[start:end, :]
        sd = beat.std(axis=0, keepdims=True)
        if (sd < 1e-8).any():
            continue
        beat = (beat - beat.mean(axis=0, keepdims=True)) / sd

        prev_rr = (r_samples[i] - r_samples[i - 1]) / FS
        next_rr = (r_samples[i + 1] - r_samples[i]) / FS
        prev2_rr = (r_samples[i - 1] - r_samples[i - 2]) / FS if i >= 2 else prev_rr
        lo, hi = max(0, i - 5), min(n, i + 6)
        local_rr = np.mean(np.diff(r_samples[lo:hi])) / FS

        feats = [
            prev_rr,
            next_rr,
            local_rr,
            prev_rr / (local_rr + eps),
            next_rr / (local_rr + eps),
            prev_rr / (rec_median_rr + eps),
            next_rr / (rec_median_rr + eps),
            prev2_rr / (local_rr + eps),
        ]
        feats[0:3] = list(np.clip(feats[0:3], 0.15, 3.0))
        feats[3:] = list(np.clip(feats[3:], 0.0, 4.0))

        beats.append(beat)
        labels.append(CLASS_TO_IDX[AAMI_MAP[symbols[i]]])
        rr_feats.append(feats)
        groups.append(record_id)

    return (
        np.array(beats, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(rr_feats, dtype=np.float32),
        np.array(groups, dtype=np.int64),
    )


def build_dataset(record_ids, dl_dir=DATA_DIR, verbose=True):
    X_list, y_list, rr_list, g_list = [], [], [], []
    for rid in record_ids:
        X, y, rr, g = extract_beats_from_record(rid, dl_dir)
        if len(X) == 0:
            continue
        X_list.append(X)
        y_list.append(y)
        rr_list.append(rr)
        g_list.append(g)
        if verbose:
            print(f"  record {rid}: {len(X)} beats")
    return (
        np.concatenate(X_list),
        np.concatenate(y_list),
        np.concatenate(rr_list),
        np.concatenate(g_list),
    )


def sparse_categorical_focal_loss(gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        probs = tf.gather(y_pred, y_true, batch_dims=1)
        focal_weight = tf.pow(1.0 - probs, gamma)
        ce = -tf.math.log(probs)
        return tf.reduce_mean(focal_weight * ce)
    return loss_fn


L2 = tf.keras.regularizers.l2(3e-5)


def transformer_block(x, num_heads=4, key_dim=16, ff_dim=128, dropout=0.2, name_prefix="tb"):
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"{name_prefix}_mha"
    )(x, x)
    attn_out = layers.Dropout(dropout)(attn_out)
    x1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
    ff = layers.Dense(ff_dim, activation="relu", kernel_regularizer=L2)(x1)
    ff = layers.Dense(x.shape[-1], kernel_regularizer=L2)(ff)
    ff = layers.Dropout(dropout)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(x1 + ff)


def build_model(total_steps, beat_len=BEAT_LEN, n_leads=N_LEADS, n_rr=N_RR, n_classes=N_CLASSES):
    beat_in = layers.Input(shape=(beat_len, n_leads), name="beat_input")

    x = layers.Conv1D(32, 7, padding="same", kernel_regularizer=L2)(beat_in)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool1D(2)(x)

    x = layers.Conv1D(64, 5, padding="same", kernel_regularizer=L2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool1D(2)(x)

    x = layers.Conv1D(64, 3, padding="same", kernel_regularizer=L2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = transformer_block(x, num_heads=4, key_dim=16, ff_dim=128, dropout=0.2, name_prefix="tb1")
    x = transformer_block(x, num_heads=4, key_dim=16, ff_dim=128, dropout=0.2, name_prefix="tb2")
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)

    rr_in = layers.Input(shape=(n_rr,), name="rr_input")
    r = layers.Dense(32, activation="relu", kernel_regularizer=L2)(rr_in)
    r = layers.Dense(16, activation="relu", kernel_regularizer=L2)(r)

    merged = layers.Concatenate()([x, r])
    merged = layers.Dense(64, activation="relu", kernel_regularizer=L2)(merged)
    merged = layers.Dropout(0.3)(merged)
    out = layers.Dense(n_classes, activation="softmax")(merged)

    model = models.Model(inputs=[beat_in, rr_in], outputs=out, name="ECG_CNN_Attention_v7")
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3, decay_steps=int(total_steps), alpha=0.02
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=sparse_categorical_focal_loss(gamma=2.0),
        metrics=["accuracy"],
    )
    return model


def apply_logit_adjustment(probs, train_priors, true_priors, tau=1.0):
    """Menon et al. (ICLR 2021): log p'_c = log p_c + tau * (log pi_true_c - log pi_train_c)."""
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    adjusted = log_probs + tau * (np.log(true_priors) - np.log(train_priors))[None, :]
    adjusted -= adjusted.max(axis=1, keepdims=True)
    e = np.exp(adjusted)
    return e / e.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    download_mitdb()
    X_train_full, y_train_full, rr_train_full, groups_train_full = build_dataset(DS1)
    X_test, y_test, rr_test, groups_test = build_dataset(DS2)

    RR_MEAN = rr_train_full.mean(axis=0)
    RR_STD = rr_train_full.std(axis=0) + 1e-6
    rr_train_full = ((rr_train_full - RR_MEAN) / RR_STD).astype(np.float32)
    rr_test = ((rr_test - RR_MEAN) / RR_STD).astype(np.float32)
    print("DS1 shape:", X_train_full.shape, "DS2 shape:", X_test.shape)
