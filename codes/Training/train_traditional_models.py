# train_classical_models.py
import os
import numpy as np
import joblib
from time import time

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from utils_data import load_tabular, fit_label_encoder, MODEL_DIR, RESULT_DIR

# ==========================================================
#                LOAD TABULAR FEATURES
# ==========================================================
X_train, X_val, X_test, y_train, y_val, y_test = load_tabular()

print("Loaded tabular shapes:")
print("  X_train:", X_train.shape)
print("  X_val:  ", X_val.shape)
print("  X_test: ", X_test.shape)

# ==========================================================
#                LABEL ENCODING
# ==========================================================
label_encoder = fit_label_encoder(y_train, y_val, y_test)

y_train_enc = label_encoder.transform(y_train)
y_val_enc   = label_encoder.transform(y_val)
y_test_enc  = label_encoder.transform(y_test)

num_classes = len(label_encoder.classes_)
print("\nClasses:", label_encoder.classes_)

# ==========================================================
#                SCALE FEATURES
# ==========================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_tabular.pkl"))
print("\nSaved scaler_tabular.pkl")

# ==========================================================
#                TRAINING FUNCTION
# ==========================================================
def evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n\n==============================")
    print(f"Training {name}...")
    print("==============================")

    start = time()
    model.fit(X_train, y_train)
    train_time = time() - start

    print(f"⏳ Train time: {train_time:.2f} sec")

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        digits=4
    )
    print(report)

    # save model
    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, model_path)

    # save report
    with open(os.path.join(RESULT_DIR, f"{name}_report.txt"), "w") as f:
        f.write(report)

    print(f"✔ Saved model → {model_path}")
    print(f"✔ Saved report → {name}_report.txt")


# ==========================================================
#                MODEL DEFINITIONS
# ==========================================================
models = {
    "logistic_regression": LogisticRegression(
        max_iter=500,
        n_jobs=-1,
        multi_class="auto"
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1
    ),

    "svm_rbf": SVC(
        kernel="rbf",
        probability=True,
        C=3,
        gamma="scale"
    ),

    "xgboost": xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist"
    )
}

# ==========================================================
#                TRAIN EACH MODEL
# ==========================================================
for name, model in models.items():
    evaluate_model(
        name,
        model,
        X_train_s, y_train_enc,
        X_val_s, y_val_enc,
        X_test_s, y_test_enc
    )

print("\n🎉 All traditional models trained successfully!")
