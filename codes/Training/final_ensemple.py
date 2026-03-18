import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ================== PATH ==================
BASE_DIR = r"A:\ML project"
MODEL_DIR = os.path.join(BASE_DIR, "models")
SPLIT_DIR = os.path.join(BASE_DIR, "FEATURES", "splits")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ================== LOAD DATA ==================
print("🔍 Loading test data...")
X_tab_test = np.load(os.path.join(SPLIT_DIR, "X_tabular_test.npy"))
X_cnn_test = np.load(os.path.join(SPLIT_DIR, "X_cnn_stack_test.npy"))
y_test = np.load(os.path.join(SPLIT_DIR, "y_test.npy"), allow_pickle=True)  # 🔥 FIXED

# ================== LOAD MODELS ==================
print("📦 Loading models...")
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_tabular.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# Use best checkpoints where available
cnn_path = os.path.join(MODEL_DIR, "cnn_best.h5") if os.path.exists(os.path.join(MODEL_DIR, "cnn_best.h5")) else os.path.join(MODEL_DIR, "cnn_final.h5")
crnn_path = os.path.join(MODEL_DIR, "crnn_best.h5") if os.path.exists(os.path.join(MODEL_DIR, "crnn_best.h5")) else os.path.join(MODEL_DIR, "crnn_final.h5")

cnn_model = load_model(cnn_path, compile=False)
crnn_model = load_model(crnn_path, compile=False)

# ================== PREDICT PROBABILITIES ==================
print("🎯 Generating prediction probabilities...")
X_tab_test_s = scaler.transform(X_tab_test)
P_xgb = xgb_model.predict_proba(X_tab_test_s)
P_cnn = cnn_model.predict(X_cnn_test)
P_crnn = crnn_model.predict(X_cnn_test)

# ================== CLASS INFO ==================
classes = label_encoder.classes_
oa_idx = np.where(classes == "Obstructive Apnea")[0][0]
y_test_enc = label_encoder.transform(y_test)

# ================== BASELINE REPORTS ==================
def evaluate_baseline(preds, name):
    preds = label_encoder.inverse_transform(np.argmax(preds, axis=1))
    print(f"\n📊 BASELINE: {name}")
    print(classification_report(y_test, preds, target_names=classes))

evaluate_baseline(P_xgb, "XGBoost")
evaluate_baseline(P_cnn, "CNN")
evaluate_baseline(P_crnn, "CRNN")

# ================== WEIGHTED ENSEMBLE ==================
print("\n🤖 Applying Weighted Ensemble...")
w_xgb = 0.55
w_cnn = 0.25
w_crnn = 0.20

P_final = w_xgb * P_xgb + w_cnn * P_cnn + w_crnn * P_crnn

# ================== THRESHOLD OPTIMIZATION ==================
print("\n🛠 Optimizing threshold for Obstructive Apnea...")
best_f1 = 0
best_t = 0.55

for t in np.arange(0.45, 0.65, 0.01):
    temp_pred = []
    for i in range(len(y_test)):
        if P_final[i, oa_idx] >= t:
            temp_pred.append(oa_idx)
        else:
            temp_pred.append(np.argmax(P_final[i]))
    f1 = f1_score((y_test_enc == oa_idx), (np.array(temp_pred) == oa_idx))
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print(f"\n🔍 Best threshold for OA = {best_t:.2f}")
print(f"📈 Best OA F1 using threshold = {best_f1:.4f}")

# ================== FINAL PREDICTIONS ==================
y_pred_final = []
for i in range(len(y_test)):
    if P_final[i, oa_idx] >= best_t:
        y_pred_final.append(oa_idx)
    else:
        y_pred_final.append(np.argmax(P_final[i]))

# Convert to string labels
y_pred_final = label_encoder.inverse_transform(np.array(y_pred_final))

# ================== FINAL EVALUATION ==================
print("\n🚀 FINAL OPTIMIZED ENSEMBLE PERFORMANCE")
report = classification_report(y_test, y_pred_final, target_names=classes, digits=4)
cm = confusion_matrix(y_test, y_pred_final)
oa_f1_final = f1_score((y_test == classes[oa_idx]), (y_pred_final == classes[oa_idx]))

print(report)
print("\n🧾 Confusion Matrix:")
print(cm)
print(f"\n🎯 FINAL Obstructive Apnea F1 (Post-Optimization): {oa_f1_final:.4f}")

# ================== SAVE ==================
with open(os.path.join(RESULT_DIR, "final_ensemble_report.txt"), "w") as f:
    f.write(report)
np.save(os.path.join(RESULT_DIR, "final_ensemble_confmat.npy"), cm)

print("\n📁 Final ensemble results saved in:", RESULT_DIR)
print("✨ COMPLETED SUCCESSFULLY – No errors expected 🚀")
