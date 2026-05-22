# =========================================
# ATS MACHINE LEARNING
# PREDIKSI KECELAKAAN KERJA
# METODE: DECISION TREE
# =========================================

# =========================================
# 1. IMPORT LIBRARY
# =========================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# 2. LOAD DATASET
# =========================================
df = pd.read_csv("dataset_kecelakaan.csv")

print("\n=== DATA AWAL ===")
print(df.head())

# =========================================
# 3. CEK DATA
# =========================================
print("\n=== NAMA KOLOM ===")
print(df.columns)

print("\n=== INFO DATA ===")
df.info()

print("\n=== MISSING VALUE ===")
print(df.isnull().sum())

print("\n=== JUMLAH DATA ===")
print("Total data:", len(df))

print("\n=== DISTRIBUSI TARGET ===")
print(df['Kecelakaan'].value_counts())

# =========================================
# 4. PREPROCESSING / FEATURE ENGINEERING
# =========================================

# Copy data
data = df.copy()

# Encoding kategori → numerik
data = pd.get_dummies(data, drop_first=True)

# Tambah fitur baru
data['Rasio_Pengalaman'] = df['Usia'] / (df['Pengalaman_Tahun'] + 1)

print("\n=== DATA SETELAH PREPROCESSING ===")
print(data.head())

# =========================================
# 5. PISAHKAN FITUR & TARGET
# =========================================
X = data.drop('Kecelakaan', axis=1)
y = data['Kecelakaan']

# =========================================
# 6. SPLIT DATA
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 7. MODEL DECISION TREE
# =========================================
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# =========================================
# 8. PREDIKSI
# =========================================
y_pred = model.predict(X_test)

# =========================================
# 9. EVALUASI
# =========================================

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\n=== CONFUSION MATRIX ===")
print(cm)

# Classification Report
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# =========================================
# 10. VISUALISASI
# =========================================
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================================
# 11. ANALISIS
# =========================================
print("\n=== ANALISIS ===")
print("""
Model digunakan untuk memprediksi apakah karyawan berisiko kecelakaan atau tidak.

Interpretasi:
- True Positive  : Prediksi benar berisiko
- True Negative  : Prediksi benar tidak berisiko
- False Positive : Prediksi salah (alarm palsu)
- False Negative : Prediksi gagal (berbahaya)

Kesimpulan:
Model Decision Tree dapat digunakan sebagai alat bantu
dalam memprediksi risiko kecelakaan kerja sehingga
perusahaan dapat melakukan tindakan pencegahan lebih awal.
""")