import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter
import json

# ---------------- PATHS ----------------
BASE_PREPRO_DIR = r"A:\ML project\final_preprocessed_events"
FEATURES_DIR = r"A:\ML project\FEATURES\numpy"
OUT_DIR = r"A:\ML project\FEATURES\splits"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- LOAD RAW FEATURE ARRAYS ----------------
X_mfcc = np.load(os.path.join(FEATURES_DIR, "X_mfcc.npy"), allow_pickle=True)
X_lfcc = np.load(os.path.join(FEATURES_DIR, "X_lfcc.npy"), allow_pickle=True)
X_logmel = np.load(os.path.join(FEATURES_DIR, "X_logmel.npy"), allow_pickle=True)
X_cqcc = np.load(os.path.join(FEATURES_DIR, "X_cqcc.npy"), allow_pickle=True)
Y_raw = np.load(os.path.join(FEATURES_DIR, "y_labels.npy"), allow_pickle=True)

print("Loaded shapes:")
print(" X_mfcc:", X_mfcc.shape)
print(" X_lfcc:", X_lfcc.shape)
print(" X_logmel:", X_logmel.shape)
print(" X_cqcc:", X_cqcc.shape)
print(" Y_raw:", Y_raw.shape)

# ---------------- SANITIZE FEATURE ARRAYS ----------------
def is_valid_feat(x):
    """Check if feature matrix is valid."""
    return (
        isinstance(x, np.ndarray)
        and x.ndim == 2
        and x.size > 10
        and not np.isnan(x).any()
        and not np.isinf(x).any()
    )

valid_mask = []
for i in range(len(X_mfcc)):
    ok = is_valid_feat(X_mfcc[i]) and is_valid_feat(X_lfcc[i]) \
         and is_valid_feat(X_logmel[i]) and is_valid_feat(X_cqcc[i])
    valid_mask.append(ok)

valid_mask = np.array(valid_mask, dtype=bool)
print(f"Valid feature matrices: {valid_mask.sum()} / {len(valid_mask)}")

# Filter out invalid rows
X_mfcc = X_mfcc[valid_mask]
X_lfcc = X_lfcc[valid_mask]
X_logmel = X_logmel[valid_mask]
X_cqcc = X_cqcc[valid_mask]
Y_raw = Y_raw[valid_mask]

# ---------------- LABEL MAPPING ----------------
def map_label(lbl):
    s = str(lbl).lower()
    if "obstruct" in s or ("apnea" in s and "central" not in s):
        return "Obstructive Apnea"
    if "hypopnea" in s:
        return "Hypopnea"
    if "snore" in s:
        return "Snore"
    if "desat" in s or "o2" in s or "spo2" in s:
        return "Desaturation"
    if "normal" in s or "breath" in s:
        return "Normal Breathing"
    return None

Y_map = np.array([map_label(y) for y in Y_raw], dtype=object)
final_mask = np.array([y is not None for y in Y_map])

print("After label filtering:", final_mask.sum())

X_mfcc = X_mfcc[final_mask]
X_lfcc = X_lfcc[final_mask]
X_logmel = X_logmel[final_mask]
X_cqcc = X_cqcc[final_mask]
Y_map = Y_map[final_mask]

# ---------------- RECONSTRUCT PATIENT LIST ----------------
patients = []
for patient in os.listdir(BASE_PREPRO_DIR):
    p = os.path.join(BASE_PREPRO_DIR, patient)
    if not os.path.isdir(p): continue
    for f in os.listdir(p):
        if f.endswith(".wav"):
            patients.append(patient)

patients = np.array(patients)[valid_mask][final_mask]

# ---------------- PATIENT SPLIT ----------------
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=32)
train_idx, temp_idx = next(gss1.split(X_mfcc, Y_map, groups=patients))

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=32)
val_rel, test_rel = next(gss2.split(X_mfcc[temp_idx], Y_map[temp_idx], groups=patients[temp_idx]))

val_idx = temp_idx[val_rel]
test_idx = temp_idx[test_rel]

print("Split sizes:", len(train_idx), len(val_idx), len(test_idx))

# ---------------- BUILD TABULAR FEATURES ----------------
def pool_features(X):
    pooled = []
    for x in X:
        mean = np.mean(x, axis=1)
        std = np.std(x, axis=1)
        pooled.append(np.concatenate([mean, std]))
    return np.vstack(pooled).astype(np.float32)

X_tab = np.hstack([
    pool_features(X_mfcc),
    pool_features(X_lfcc),
    pool_features(X_cqcc),
    pool_features(X_logmel),
])

# CNN inputs
X_cnn_mfcc = np.stack(X_mfcc).astype(np.float32)  # (N, F, T)
X_cnn_logmel = np.stack(X_logmel).astype(np.float32)

# Pad frequency axis to match
freq_dim = max(X_cnn_mfcc.shape[1], X_cnn_logmel.shape[1])

def pad_freq(a, target):
    if a.shape[1] < target:
        pad = target - a.shape[1]
        return np.pad(a, ((0,0),(0,pad),(0,0)))
    return a[:, :target]

X_mfcc_p = pad_freq(X_cnn_mfcc, freq_dim)
X_logmel_p = pad_freq(X_cnn_logmel, freq_dim)

# stack as channels
X_cnn_stack = np.stack([X_mfcc_p, X_logmel_p], axis=-1)

# ---------------- SAVE SPLITS ----------------
splits = {
    "train": train_idx,
    "val": val_idx,
    "test": test_idx
}

for name, idx in splits.items():
    np.save(os.path.join(OUT_DIR, f"X_tabular_{name}.npy"), X_tab[idx])
    np.save(os.path.join(OUT_DIR, f"X_cnn_mfcc_{name}.npy"), X_cnn_mfcc[idx])
    np.save(os.path.join(OUT_DIR, f"X_cnn_stack_{name}.npy"), X_cnn_stack[idx])
    np.save(os.path.join(OUT_DIR, f"y_{name}.npy"), Y_map[idx])
    print(f"{name} class counts:", Counter(Y_map[idx]))

with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump({
        "counts": dict(Counter(Y_map)),
        "splits": {k: len(v) for k, v in splits.items()},
        "classes": sorted(list(set(Y_map)))
    }, f, indent=2)

print("\n✔️ All splits saved successfully!")
