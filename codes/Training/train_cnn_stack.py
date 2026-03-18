# train_cnn_stack.py
import os, numpy as np, joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from utils_data import load_cnn_stack, fit_label_encoder, MODEL_DIR, RESULT_DIR

# -------- LOAD DATA --------
X_train, X_val, X_test, y_train, y_val, y_test = load_cnn_stack()
print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

# -------- LABEL ENCODER --------
le = fit_label_encoder(y_train, y_val, y_test)
y_train_enc = le.transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)
num_classes = len(le.classes_)
y_train_oh = to_categorical(y_train_enc, num_classes)
y_val_oh   = to_categorical(y_val_enc, num_classes)

# -------- BUILD MODEL --------
input_shape = X_train.shape[1:]  # (freq, time, channels)
def build_cnn(inp_shape, n_classes):
    inp = Input(shape=inp_shape)
    x = Conv2D(32, (3,3), padding="same")(inp); x = BatchNormalization()(x); x = Activation("relu")(x)
    x = MaxPool2D((2,2))(x)
    x = Conv2D(64, (3,3), padding="same")(x); x = BatchNormalization()(x); x = Activation("relu")(x)
    x = MaxPool2D((2,2))(x)
    x = Conv2D(128, (3,3), padding="same")(x); x = BatchNormalization()(x); x = Activation("relu")(x)
    x = MaxPool2D((2,2))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    out = Dense(n_classes, activation="softmax")(x)
    return Model(inp, out)

model = build_cnn(input_shape, num_classes)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# -------- CALLBACKS & TRAIN --------
os.makedirs(MODEL_DIR, exist_ok=True)
ckpt = ModelCheckpoint(os.path.join(MODEL_DIR, "cnn_best.h5"), monitor="val_loss", save_best_only=True)
es = EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")
rlr = ReduceLROnPlateau(patience=4, factor=0.5, monitor="val_loss")

history = model.fit(
    X_train, y_train_oh,
    validation_data=(X_val, y_val_oh),
    epochs=60, batch_size=64,
    callbacks=[ckpt, es, rlr]
)

model.save(os.path.join(MODEL_DIR, "cnn_final.h5"))
joblib.dump(history.history, os.path.join(RESULT_DIR, "cnn_history.pkl"))

# -------- EVALUATE ON TEST --------
from sklearn.metrics import classification_report, confusion_matrix
y_pred_prob = model.predict(X_test)
y_pred = y_pred_prob.argmax(axis=1)
report = classification_report(y_test_enc, y_pred, target_names=le.classes_, digits=4)
cm = confusion_matrix(y_test_enc, y_pred)

with open(os.path.join(RESULT_DIR, "cnn_test_report.txt"), "w") as f:
    f.write(report)
np.save(os.path.join(RESULT_DIR, "cnn_confmat.npy"), cm)
print(report)
print("CNN saved to:", MODEL_DIR)
