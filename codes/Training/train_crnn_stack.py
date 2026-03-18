# train_crnn_stack.py
import os, numpy as np, joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Permute, Reshape, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from utils_data import load_cnn_stack, fit_label_encoder, MODEL_DIR, RESULT_DIR

# load
X_train, X_val, X_test, y_train, y_val, y_test = load_cnn_stack()
print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

# label encoder
le = fit_label_encoder(y_train, y_val, y_test)
y_train_enc = le.transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)
num_classes = len(le.classes_)
y_train_oh = to_categorical(y_train_enc, num_classes)
y_val_oh   = to_categorical(y_val_enc, num_classes)

# build CRNN: conv layers that pool only freq dimension, keep time steps
freq, time_steps, channels = X_train.shape[1], X_train.shape[2], X_train.shape[3]
inp = Input(shape=(freq, time_steps, channels))

x = Conv2D(32, (3,3), padding="same")(inp); x = BatchNormalization()(x); x = Activation("relu")(x)
x = MaxPool2D((2,1))(x)   # pool freq, preserve time
x = Conv2D(64, (3,3), padding="same")(x); x = BatchNormalization()(x); x = Activation("relu")(x)
x = MaxPool2D((2,1))(x)

# permute to (time, freq', channels) then reshape to (time, features)
x = Permute((2,1,3))(x)
shape = x.shape
x = Reshape((int(shape[1]), int(shape[2])*int(shape[3])))(x)

x = Bidirectional(LSTM(128, return_sequences=False))(x)
x = Dropout(0.4)(x)
out = Dense(num_classes, activation="softmax")(x)

model = Model(inp, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# callbacks
ckpt = ModelCheckpoint(os.path.join(MODEL_DIR, "crnn_best.h5"), monitor="val_loss", save_best_only=True)
es = EarlyStopping(patience=8, restore_best_weights=True)
rlr = ReduceLROnPlateau(patience=4, factor=0.5)

history = model.fit(X_train, y_train_oh, validation_data=(X_val, y_val_oh),
                    epochs=60, batch_size=32, callbacks=[ckpt, es, rlr])

model.save(os.path.join(MODEL_DIR, "crnn_final.h5"))
joblib.dump(history.history, os.path.join(RESULT_DIR, "crnn_history.pkl"))

# evaluate
from sklearn.metrics import classification_report, confusion_matrix
y_pred_prob = model.predict(X_test)
y_pred = y_pred_prob.argmax(axis=1)
report = classification_report(y_test_enc, y_pred, target_names=le.classes_, digits=4)
cm = confusion_matrix(y_test_enc, y_pred)

with open(os.path.join(RESULT_DIR, "crnn_test_report.txt"), "w") as f:
    f.write(report)
np.save(os.path.join(RESULT_DIR, "crnn_confmat.npy"), cm)
print(report)
print("CRNN saved to:", MODEL_DIR)
