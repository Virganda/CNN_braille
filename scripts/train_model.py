"""
Train the Braille CNN model using the labeled dataset in data/train and data/test.

Usage:
    python scripts/train_model.py

Output:
    braille_cnn.h5       — trained model weights
    model_labels.txt     — class label list (order matches model output)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from cnn_model import create_model

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DIR  = 'data/train'
TEST_DIR   = 'data/test'
IMG_SIZE   = (28, 28)
BATCH_SIZE = 32
EPOCHS     = 50
MODEL_OUT  = 'braille_cnn.h5'
LABELS_OUT = 'model_labels.txt'
# ─────────────────────────────────────────────────────────────────────────────


def find_csv(directory):
    for f in os.listdir(directory):
        if f.endswith('.csv'):
            return os.path.join(directory, f)
    raise FileNotFoundError(f"No CSV found in {directory}")


def load_dataset(directory, label_cols):
    """
    Reads the CSV in `directory`, loads each image, returns (X, y) arrays.
    Images are resized to IMG_SIZE and normalised to [0, 1].
    Labels are one-hot encoded using `label_cols` order.
    """
    csv_path = find_csv(directory)
    df = pd.read_csv(csv_path)

    # Strip whitespace from column names (the CSV has spaces after commas)
    df.columns = [c.strip() for c in df.columns]
    df['filename'] = df['filename'].str.strip()

    X, y = [], []
    skipped = 0

    for _, row in df.iterrows():
        img_path = os.path.join(directory, row['filename'])
        if not os.path.exists(img_path):
            skipped += 1
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        img = cv2.resize(img, IMG_SIZE)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)   # (28, 28, 1)

        # One-hot label vector
        label_vec = row[label_cols].values.astype('float32')
        # If all zeros (unlabeled), skip
        if label_vec.sum() == 0:
            skipped += 1
            continue

        X.append(img)
        y.append(label_vec)

    print(f"  Loaded {len(X)} samples from {directory} (skipped {skipped})")
    return np.array(X), np.array(y)


def main():
    # ── Discover labels from CSV header ──────────────────────────────────────
    csv_path = find_csv(TRAIN_DIR)
    df_header = pd.read_csv(csv_path, nrows=0)
    df_header.columns = [c.strip() for c in df_header.columns]

    # All columns except 'filename' are class labels
    label_cols = [c for c in df_header.columns if c != 'filename']
    num_classes = len(label_cols)
    print(f"Classes ({num_classes}): {label_cols}")

    # Save label order so app.py can map predictions back to characters
    with open(LABELS_OUT, 'w') as f:
        f.write('\n'.join(label_cols))
    print(f"Saved label list → {LABELS_OUT}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading training data...")
    X_train, y_train = load_dataset(TRAIN_DIR, label_cols)

    print("Loading test/validation data...")
    X_test, y_test = load_dataset(TEST_DIR, label_cols)

    print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")

    # ── Class weights (handle imbalanced data) ────────────────────────────────
    y_int = np.argmax(y_train, axis=1)
    classes = np.unique(y_int)
    cw = compute_class_weight('balanced', classes=classes, y=y_int)
    class_weight_dict = dict(zip(classes, cw))

    # ── Build model ───────────────────────────────────────────────────────────
    model = create_model(num_classes=num_classes)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_accuracy', verbose=1),
        EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, monitor='val_loss', verbose=1),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal test accuracy: {acc * 100:.2f}%  |  loss: {loss:.4f}")
    print(f"Model saved → {MODEL_OUT}")


if __name__ == '__main__':
    main()
