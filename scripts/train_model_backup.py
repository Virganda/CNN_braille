"""
Train the Braille CNN model using the labeled dataset in data/train and data/test.

Backup version with:
✔ Accuracy & Loss Graph
✔ Safe backup model output
✔ Same architecture as original
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.utils.class_weight import compute_class_weight

from cnn_model import create_model


# ───────────────── CONFIG ─────────────────

TRAIN_DIR  = 'data/train'
TEST_DIR   = 'data/test'

IMG_SIZE   = (28, 28)

BATCH_SIZE = 32
EPOCHS     = 50

# SAFE OUTPUT (biar model lama aman huhu)
MODEL_OUT  = 'braille_cnn_backup.h5'

LABELS_OUT = 'model_labels.txt'

# ──────────────────────────────────────────


def find_csv(directory):
    for f in os.listdir(directory):
        if f.endswith('.csv'):
            return os.path.join(directory, f)

    raise FileNotFoundError(f"No CSV found in {directory}")


def load_dataset(directory, label_cols):

    csv_path = find_csv(directory)

    df = pd.read_csv(csv_path)

    # hapus spasi kolom
    df.columns = [c.strip() for c in df.columns]

    # rapihin filename
    df['filename'] = df['filename'].str.strip()

    X = []
    y = []

    skipped = 0

    for _, row in df.iterrows():

        img_path = os.path.join(directory, row['filename'])

        # skip kalau file tidak ada
        if not os.path.exists(img_path):
            skipped += 1
            continue

        # grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            skipped += 1
            continue

        # resize
        img = cv2.resize(img, IMG_SIZE)

        # normalisasi
        img = img.astype('float32') / 255.0

        # tambah dimensi channel
        img = np.expand_dims(img, axis=-1)

        # label one-hot
        label_vec = row[label_cols].values.astype('float32')

        # skip unlabeled
        if label_vec.sum() == 0:
            skipped += 1
            continue

        X.append(img)
        y.append(label_vec)

    print(f"Loaded {len(X)} samples from {directory}")
    print(f"Skipped {skipped} samples")

    return np.array(X), np.array(y)


def main():

    # ───────────────── LABEL ─────────────────
    #pembacaan label
    csv_path = find_csv(TRAIN_DIR)

    df_header = pd.read_csv(csv_path, nrows=0)

    df_header.columns = [c.strip() for c in df_header.columns]

    # semua kolom selain filename
    label_cols = [c for c in df_header.columns if c != 'filename']

    num_classes = len(label_cols)

    print(f"\nDetected {num_classes} classes")
    print(label_cols)

    # save label
    with open(LABELS_OUT, 'w') as f:
        f.write('\n'.join(label_cols))

    print(f"\nSaved labels -> {LABELS_OUT}")

    # ───────────────── LOAD DATA ─────────────────
    #pembacaan data
    print("\nLoading training data...")
    X_train, y_train = load_dataset(TRAIN_DIR, label_cols)

    print("\nLoading validation data...")
    X_test, y_test = load_dataset(TEST_DIR, label_cols)

    print(f"\nTrain Shape : {X_train.shape}")
    print(f"Test Shape  : {X_test.shape}")

    # ───────────────── CLASS WEIGHT ─────────────────

    y_int = np.argmax(y_train, axis=1)

    classes = np.unique(y_int)
    #PENYEIMBANGAN DATA
    cw = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_int
    )

    class_weight_dict = dict(zip(classes, cw))

    # ───────────────── MODEL ─────────────────
    #PEMANGGILAN SANG CNN
    model = create_model(num_classes=num_classes)

    print("\nMODEL SUMMARY")
    model.summary()

    # ───────────────── CALLBACKS ─────────────────

    callbacks = [

        ModelCheckpoint(
            MODEL_OUT,
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        ),

        EarlyStopping(
            patience=8,
            restore_best_weights=True,
            monitor='val_accuracy'
        ),

        ReduceLROnPlateau(
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            monitor='val_loss',
            verbose=1
        )
    ]

    # ───────────────── TRAINING ─────────────────

    print("\nStarting training...\n")

    history = model.fit(

        X_train,
        y_train,

        validation_data=(X_test, y_test),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weight_dict,

        callbacks=callbacks,

        verbose=1
    )

    # ───────────────── EVALUATION ─────────────────

    print("\nEvaluating model...\n")

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Final Accuracy : {acc * 100:.2f}%")
    print(f"Final Loss     : {loss:.4f}")

    print(f"\nModel saved -> {MODEL_OUT}")

    # ───────────────── GRAFIK ACCURACY ─────────────────

    plt.figure(figsize=(8,5))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.title('Model Accuracy')

    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.legend(['Train', 'Validation'])

    plt.grid(True)

    plt.show()

    # ───────────────── GRAFIK LOSS ─────────────────

    plt.figure(figsize=(8,5))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('Model Loss')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Train', 'Validation'])

    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    main()