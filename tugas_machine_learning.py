# ============================================================
# TUGAS MACHINE LEARNING DASAR
# Prediksi Kelulusan Mahasiswa Tepat Waktu
# ============================================================

# ============================================================
# 1. IMPORT LIBRARY
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 2. MEMBACA DATASET
# ============================================================

df = pd.read_csv("mahasiswa_lulus.csv")

print("==========================================")
print("DATASET")
print("==========================================")

print("\n5 Data Pertama:")
print(df.head())

print("\nUkuran Dataset:")
print(df.shape)


# ============================================================
# 3. PREPROCESSING DATA
# ============================================================

# Mengecek missing value
print("\nJumlah Missing Value Sebelum Preprocessing:")
print(df.isnull().sum())


# Menentukan kolom numerik dan kategorikal
kolom_numerik = [
    "IPK",
    "Kehadiran",
    "Jam_Belajar"
]

kolom_kategorikal = [
    "Organisasi",
    "Penghasilan_Ortu",
    "Jenis_Kelamin",
    "Status_Beasiswa"
]


# Menangani missing value pada data numerik
# Menggunakan median
for kolom in kolom_numerik:
    df[kolom] = df[kolom].fillna(df[kolom].median())


# Menangani missing value pada data kategorikal
# Menggunakan modus
for kolom in kolom_kategorikal:
    df[kolom] = df[kolom].fillna(df[kolom].mode()[0])


print("\nJumlah Missing Value Setelah Preprocessing:")
print(df.isnull().sum())


# ============================================================
# 4. ENCODING DATA KATEGORIKAL
# ============================================================

encoder = LabelEncoder()

kolom_encoding = [
    "Organisasi",
    "Penghasilan_Ortu",
    "Jenis_Kelamin",
    "Status_Beasiswa",
    "Lulus_Tepat_Waktu"
]

for kolom in kolom_encoding:
    df[kolom] = encoder.fit_transform(df[kolom])


print("\nData Setelah Encoding:")
print(df.head())


# ============================================================
# 5. MEMISAHKAN FITUR DAN TARGET
# ============================================================

# X = fitur yang digunakan untuk prediksi
X = df.drop("Lulus_Tepat_Waktu", axis=1)

# y = target yang ingin diprediksi
y = df["Lulus_Tepat_Waktu"]


print("\nFitur yang digunakan:")
print(X.columns)

print("\nJumlah Data Setiap Kelas:")
print(y.value_counts())


# ============================================================
# 6. TRAIN-TEST SPLIT 80:20
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nJumlah Data Training:", len(X_train))
print("Jumlah Data Testing :", len(X_test))


# ============================================================
# 7. PEMBUATAN MODEL
# ============================================================

# Decision Tree dengan kedalaman maksimum 3
model_dt_3 = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)


# Decision Tree tanpa batas kedalaman
model_dt_none = DecisionTreeClassifier(
    max_depth=None,
    random_state=42
)


# Gaussian Naive Bayes
model_nb = GaussianNB()


# ============================================================
# 8. TRAINING MODEL
# ============================================================

model_dt_3.fit(X_train, y_train)
model_dt_none.fit(X_train, y_train)
model_nb.fit(X_train, y_train)

print("\nKetiga model berhasil dilatih.")


# ============================================================
# 9. PREDIKSI DATA TESTING
# ============================================================

y_pred_dt_3 = model_dt_3.predict(X_test)
y_pred_dt_none = model_dt_none.predict(X_test)
y_pred_nb = model_nb.predict(X_test)


# ============================================================
# 10. EVALUASI MODEL
# ============================================================

def hitung_metrik(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    return accuracy, precision, recall, f1


# Menghitung metrik Decision Tree max_depth=3
acc_dt3, pre_dt3, rec_dt3, f1_dt3 = hitung_metrik(
    y_test,
    y_pred_dt_3
)


# Menghitung metrik Decision Tree max_depth=None
acc_dt_none, pre_dt_none, rec_dt_none, f1_dt_none = hitung_metrik(
    y_test,
    y_pred_dt_none
)


# Menghitung metrik Gaussian Naive Bayes
acc_nb, pre_nb, rec_nb, f1_nb = hitung_metrik(
    y_test,
    y_pred_nb
)


# Membuat tabel hasil evaluasi
hasil_evaluasi = pd.DataFrame({
    "Model": [
        "Decision Tree (max_depth=3)",
        "Decision Tree (max_depth=None)",
        "Gaussian Naive Bayes"
    ],
    "Accuracy": [
        acc_dt3,
        acc_dt_none,
        acc_nb
    ],
    "Precision": [
        pre_dt3,
        pre_dt_none,
        pre_nb
    ],
    "Recall": [
        rec_dt3,
        rec_dt_none,
        rec_nb
    ],
    "F1-Score": [
        f1_dt3,
        f1_dt_none,
        f1_nb
    ]
})


hasil_evaluasi = hasil_evaluasi.round(4)


print("\n==========================================")
print("HASIL EVALUASI MODEL")
print("==========================================")

print(hasil_evaluasi.to_string(index=False))


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm_dt3 = confusion_matrix(
    y_test,
    y_pred_dt_3
)

cm_dt_none = confusion_matrix(
    y_test,
    y_pred_dt_none
)

cm_nb = confusion_matrix(
    y_test,
    y_pred_nb
)


print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")


print("\nDecision Tree (max_depth=3):")
print(cm_dt3)


print("\nDecision Tree (max_depth=None):")
print(cm_dt_none)


print("\nGaussian Naive Bayes:")
print(cm_nb)


# ============================================================
# 12. VISUALISASI DECISION TREE
# ============================================================

plt.figure(figsize=(20, 10))

plot_tree(
    model_dt_3,
    feature_names=X.columns,
    class_names=["Tidak", "Ya"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree - max_depth=3")
plt.tight_layout()

plt.savefig(
    "hasil/decision_tree_max_depth_3.png",
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 13. BAR CHART PERBANDINGAN METRIK
# ============================================================

nama_model = [
    "DT max_depth=3",
    "DT max_depth=None",
    "GaussianNB"
]

accuracy = [
    acc_dt3,
    acc_dt_none,
    acc_nb
]

precision = [
    pre_dt3,
    pre_dt_none,
    pre_nb
]

recall = [
    rec_dt3,
    rec_dt_none,
    rec_nb
]

f1_score_data = [
    f1_dt3,
    f1_dt_none,
    f1_nb
]


x = np.arange(len(nama_model))
lebar = 0.2


plt.figure(figsize=(12, 6))

plt.bar(
    x - 1.5 * lebar,
    accuracy,
    width=lebar,
    label="Accuracy"
)

plt.bar(
    x - 0.5 * lebar,
    precision,
    width=lebar,
    label="Precision"
)

plt.bar(
    x + 0.5 * lebar,
    recall,
    width=lebar,
    label="Recall"
)

plt.bar(
    x + 1.5 * lebar,
    f1_score_data,
    width=lebar,
    label="F1-Score"
)


plt.xlabel("Model")
plt.ylabel("Nilai Metrik")
plt.title("Perbandingan Kinerja 3 Model")
plt.xticks(x, nama_model)
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()

plt.savefig(
    "hasil/perbandingan_metrik.png",
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 14. VISUALISASI CONFUSION MATRIX
# ============================================================

model_confusion = [
    (
        "Decision Tree max_depth=3",
        cm_dt3,
        "hasil/confusion_matrix_dt3.png"
    ),
    (
        "Decision Tree max_depth=None",
        cm_dt_none,
        "hasil/confusion_matrix_dt_none.png"
    ),
    (
        "Gaussian Naive Bayes",
        cm_nb,
        "hasil/confusion_matrix_nb.png"
    )
]


for nama, cm, nama_file in model_confusion:

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title(f"Confusion Matrix - {nama}")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")

    plt.xticks(
        [0, 1],
        ["Tidak", "Ya"]
    )

    plt.yticks(
        [0, 1],
        ["Tidak", "Ya"]
    )


    # Menampilkan nilai pada setiap kotak
    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=14
            )


    plt.colorbar()
    plt.tight_layout()

    plt.savefig(
        nama_file,
        dpi=300
    )

    plt.show()
    plt.close()


# ============================================================
# 15. ANALISIS 5 DATA YANG SALAH DIPREDIKSI
# ============================================================

data_analisis = X_test.copy()

data_analisis["Aktual"] = y_test
data_analisis["Prediksi"] = y_pred_nb


# Mengambil data yang salah diprediksi
data_salah = data_analisis[
    data_analisis["Aktual"] != data_analisis["Prediksi"]
]


print("\n==========================================")
print("5 DATA YANG SALAH DIPREDIKSI")
print("==========================================")

print(
    data_salah.head(5).to_string()
)


# ============================================================
# 16. SIMULASI MAHASISWA
# ============================================================

# Contoh mahasiswa:
# IPK tinggi tetapi kehadiran rendah

data_simulasi = pd.DataFrame({
    "IPK": [3.80],
    "Kehadiran": [60.0],
    "Jam_Belajar": [4.0],
    "Organisasi": [1],
    "Penghasilan_Ortu": [1],
    "Jenis_Kelamin": [0],
    "Status_Beasiswa": [0]
})


# Prediksi menggunakan Decision Tree
prediksi_dt3 = model_dt_3.predict(
    data_simulasi
)[0]


# Prediksi menggunakan Gaussian Naive Bayes
prediksi_nb = model_nb.predict(
    data_simulasi
)[0]


print("\n==========================================")
print("SIMULASI MAHASISWA")
print("==========================================")


print(
    "IPK              :",
    data_simulasi["IPK"][0]
)

print(
    "Kehadiran        :",
    data_simulasi["Kehadiran"][0],
    "%"
)

print(
    "Jam Belajar      :",
    data_simulasi["Jam_Belajar"][0],
    "jam"
)

print(
    "Organisasi       :",
    data_simulasi["Organisasi"][0]
)

print(
    "Penghasilan Ortu :",
    data_simulasi["Penghasilan_Ortu"][0]
)

print(
    "Jenis Kelamin    :",
    data_simulasi["Jenis_Kelamin"][0]
)

print(
    "Status Beasiswa  :",
    data_simulasi["Status_Beasiswa"][0]
)


print(
    "\nPrediksi Decision Tree max_depth=3 :",
    "Ya" if prediksi_dt3 == 1 else "Tidak"
)

print(
    "Prediksi Gaussian Naive Bayes     :",
    "Ya" if prediksi_nb == 1 else "Tidak"
)


# ============================================================
# SELESAI
# ============================================================

print("\n==========================================")
print("PROGRAM SELESAI")
print("==========================================")