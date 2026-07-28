import os
import numpy as np
import pandas as pd
import cv2

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD MODEL
# =========================
# PERSIAPAN DATA UJI
model = load_model("braille_cnn.h5")

# =========================
# LOAD LABEL
# =========================

with open("model_labels.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

# hanya huruf a-z
valid_labels = list("abcdefghijklmnopqrstuvwxyz")

# =========================
# LOAD CSV
# =========================

df = pd.read_csv("data/test/_classes.csv")

# hapus spasi nama kolom
df.columns = df.columns.str.strip()

y_true = []
y_pred = []

# =========================
# PREDIKSI DATA TEST
# =========================

for _, row in df.iterrows():

    filename = row['filename']

    # ambil label asli
    label = row[1:].idxmax().strip()

    # hanya huruf a-z
    if label not in valid_labels:
        continue

    img_path = os.path.join("data/test", filename)

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)

    if img is None:
        continue

    # preprocessing
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype('float32') / 255.0

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # PREDIKSI SATU SATU YAAWW
    pred = model.predict(img, verbose=0)

    pred_label = class_names[np.argmax(pred)]
    #PENGUMPULAN JAWABAN
    y_true.append(label)
    y_pred.append(pred_label)

# =========================
# EVALUASI
# =========================

if len(y_true) == 0:

    print("❌ Tidak ada data yang berhasil diproses.")
    print("Cek lagi CSV / label / path.")

else:

    print("=" * 50)
    print("HASIL EVALUASI MODEL")
    print("=" * 50)

    print("\nJumlah data diuji :", len(y_true))
    #AKURASI GLOBAL
    acc = accuracy_score(y_true, y_pred)

    print("\nAccuracy :", round(acc * 100, 2), "%")

    # =========================
    # CONFUSION MATRIX
    # =========================

    labels = sorted(list(set(y_true)))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print("\nConfusion Matrix:\n")
    print(cm)

    # =========================
    # SIMPAN GAMBAR CM
    # =========================

    plt.figure(figsize=(16, 14))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title("Confusion Matrix CNN Braille")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png",
        dpi=300
    )

    plt.show()

    print("\n✅ Confusion Matrix disimpan sebagai:")
    print("confusion_matrix.png")

    # =========================
    # CLASSIFICATION REPORT
    # =========================

    print("\nClassification Report:\n")

    report = classification_report(
        y_true,
        y_pred
    )

    print(report)

    # =========================
    # DETAIL PERHITUNGAN MANUAL
    # SEMUA HURUF (A-Z)
    # =========================

    print("\n" + "=" * 50)
    print("DETAIL PERHITUNGAN TP, FP, FN, TN SETIAP HURUF")
    print("=" * 50)

    for huruf in labels:
        
        idx = labels.index(huruf)

        TP = cm[idx, idx]
        FP = cm[:, idx].sum() - TP
        FN = cm[idx, :].sum() - TP
        TN = cm.sum() - TP - FP - FN

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(f"\nHuruf '{huruf.upper()}':")
        print(f"TP = {TP} | FP = {FP} | FN = {FN} | TN = {TN}")
        print(f"Precision = {round(precision, 4)}")
        print(f"Recall    = {round(recall, 4)}")
        print(f"F1-Score  = {round(f1, 4)}")