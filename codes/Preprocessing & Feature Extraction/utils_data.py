# utils_data.py
import os, numpy as np, joblib
from sklearn.preprocessing import LabelEncoder

SPLIT_DIR = r"A:\ML project\FEATURES\splits"
MODEL_DIR = r"A:\ML project\models"
RESULT_DIR = r"A:\ML project\results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def load_cnn_stack():
    X_train = np.load(os.path.join(SPLIT_DIR, "X_cnn_stack_train.npy"))
    X_val   = np.load(os.path.join(SPLIT_DIR, "X_cnn_stack_val.npy"))
    X_test  = np.load(os.path.join(SPLIT_DIR, "X_cnn_stack_test.npy"))
    y_train = np.load(os.path.join(SPLIT_DIR, "y_train.npy"), allow_pickle=True)
    y_val   = np.load(os.path.join(SPLIT_DIR, "y_val.npy"), allow_pickle=True)
    y_test  = np.load(os.path.join(SPLIT_DIR, "y_test.npy"), allow_pickle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_tabular():
    X_train = np.load(os.path.join(SPLIT_DIR, "X_tabular_train.npy"))
    X_val   = np.load(os.path.join(SPLIT_DIR, "X_tabular_val.npy"))
    X_test  = np.load(os.path.join(SPLIT_DIR, "X_tabular_test.npy"))
    y_train = np.load(os.path.join(SPLIT_DIR, "y_train.npy"), allow_pickle=True)
    y_val   = np.load(os.path.join(SPLIT_DIR, "y_val.npy"), allow_pickle=True)
    y_test  = np.load(os.path.join(SPLIT_DIR, "y_test.npy"), allow_pickle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test

def fit_label_encoder(y_train, y_val, y_test):
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val, y_test]))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return le

def load_label_encoder():
    return joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
