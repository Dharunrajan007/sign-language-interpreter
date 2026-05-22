#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
retrain.py

Retrains the keypoint classifier with all 10 gesture classes
and exports a new keypoint_classifier.tflite model.

Run AFTER collect_data.py has collected enough samples.
"""

import numpy as np
import tensorflow as tf
import csv
import os
from sklearn.model_selection import train_test_split

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, 'model', 'keypoint_classifier', 'keypoint.csv')
MODEL_SAVE = os.path.join(BASE_DIR, 'model', 'keypoint_classifier', 'keypoint_classifier.hdf5')
TFLITE_OUT = os.path.join(BASE_DIR, 'model', 'keypoint_classifier', 'keypoint_classifier.tflite')
LABELS     = ['HELP', 'PAIN', 'WATER', 'DOCTOR', 'EMERGENCY', 'OK', 'STOP', 'CALL', 'MEDICINE', 'BREATH']
NUM_CLASSES = len(LABELS)

#  Load dataset 
print(" Loading dataset...")
X, y = [], []
with open(CSV_PATH, encoding='utf-8-sig') as f:
    for row in csv.reader(f):
        if not row:
            continue
        label = int(row[0])
        features = [float(v) for v in row[1:]]
        if len(features) == 42:   # 21 landmarks  2 (x,y)
            X.append(features)
            y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

from collections import Counter
dist = Counter(y)
print(" Class distribution:")
for idx, name in enumerate(LABELS):
    cnt = dist.get(idx, 0)
    bar = '' * (cnt // 50)
    print(f"  [{idx}] {name:12s}: {cnt:5d}  {bar}")

# Filter: only keep classes that have data
valid_classes = sorted([idx for idx in dist if dist[idx] > 0])
mask = np.isin(y, valid_classes)
X, y = X[mask], y[mask]

# Remap labels to 0..N-1 if some classes are missing
label_map = {old: new for new, old in enumerate(valid_classes)}
y_remapped = np.array([label_map[lbl] for lbl in y])
num_classes_actual = len(valid_classes)

print(f"\n Using {num_classes_actual} classes, {len(X)} total samples")

#  Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y_remapped, test_size=0.2, random_state=42, stratify=y_remapped
)

#  Build model 
print("\n  Building model...")
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(42,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes_actual, activation='softmax'),
])
model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

#  Train 
print("\n Training...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
]
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1,
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n Test Accuracy: {acc*100:.2f}%  |  Loss: {loss:.4f}")

#  Save HDF5 
model.save(MODEL_SAVE)
print(f" Saved model: {MODEL_SAVE}")

#  Export TFLite 
print(" Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(TFLITE_OUT, 'wb') as f:
    f.write(tflite_model)
print(f" TFLite model saved: {TFLITE_OUT}")

#  Update label CSV if needed 
label_csv = os.path.join(BASE_DIR, 'model', 'keypoint_classifier', 'keypoint_classifier_label.csv')
actual_labels = [LABELS[i] for i in valid_classes]
with open(label_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    for lbl in actual_labels:
        writer.writerow([lbl])
print(f"  Labels updated: {actual_labels}")

print("\n Retraining complete! Restart server.py to use the new model.")
