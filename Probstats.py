import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import gaussian_kde

file_path = "Jawaban Survei - Form Responses 1.csv"

data = pd.read_csv(file_path)

col_status = data.columns[3]   # Status keaktifan

col_ipk = data.columns[4]      # Rentang IPK

data = data[[col_status, col_ipk]].dropna()

data["Status"] = data[col_status].str.strip()

data["IPK_Rentang"] = data[col_ipk].str.strip()

urutan_ipk = ["2.50-2.99", "3.00-3.50", ">3.50"]

tabel_kontingensi = pd.crosstab(

    data["Status"],

    data["IPK_Rentang"]

).reindex(columns=urutan_ipk, fill_value=0)

print("\n=== TABEL KONTINGENSI ===")

print(tabel_kontingensi)

aktif = tabel_kontingensi.loc["Aktif"].values

tidak = tabel_kontingensi.loc["Tidak Aktif"].values

x = np.arange(len(urutan_ipk))

width = 0.35

plt.figure(figsize=(9,6))

plt.bar(x - width/2, aktif, width, label="Aktif")

plt.bar(x + width/2, tidak, width, label="Tidak Aktif")

plt.xticks(x, urutan_ipk)

plt.xlabel("Rentang IPK")

plt.ylabel("Jumlah Mahasiswa")

plt.title("Distribusi IPK Mahasiswa\nAktif vs Tidak Aktif Organisasi")

plt.legend()

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.show()

ipk_map = {

    "2.50-2.99": 0,

    "3.00-3.50": 1,

    ">3.50": 2

}

status_map = {"Aktif": 0, "Tidak Aktif": 1}

data["IPK_Num"] = data["IPK_Rentang"].map(ipk_map)

data["Status_Num"] = data["Status"].map(status_map)

plt.figure(figsize=(8,6))

plt.scatter(

    data["Status_Num"] + np.random.uniform(-0.05, 0.05, len(data)),

    data["IPK_Num"] + np.random.uniform(-0.05, 0.05, len(data)),

    alpha=0.6

)

plt.xticks([0,1], ["Aktif", "Tidak Aktif"])

plt.yticks([0,1,2], urutan_ipk)

plt.xlabel("Status Keaktifan")

plt.ylabel("Rentang IPK")

plt.title("Scatter Plot IPK vs Status Organisasi")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()
frekuensi_ipk = (

    data["IPK_Rentang"]

    .value_counts()

    .reindex(urutan_ipk, fill_value=0)

)

nilai_ipk = {

    "2.50-2.99": 2.75,

    "3.00-3.50": 3.25,

    ">3.50": 3.75

}

x_mid = np.array([nilai_ipk[k] for k in frekuensi_ipk.index])

f = frekuensi_ipk.values

mean_ipk = np.sum(f * x_mid) / np.sum(f)

kumulatif = np.cumsum(f)

n = np.sum(f)

median_class = frekuensi_ipk.index[kumulatif >= n/2][0]

median_ipk = nilai_ipk[median_class]

modus_class = frekuensi_ipk.idxmax()

modus_ipk = nilai_ipk[modus_class]

mean_pos = np.interp(mean_ipk, x_mid, np.arange(len(x_mid)))

median_pos = urutan_ipk.index(median_class)

modus_pos = urutan_ipk.index(modus_class)

plt.figure(figsize=(9,6))

plt.bar(urutan_ipk, frekuensi_ipk.values, edgecolor="black", alpha=0.7)

plt.axvline(mean_pos, linestyle="--", linewidth=2, label=f"Mean ({mean_ipk:.2f})")

plt.axvline(median_pos, linestyle="-.", linewidth=2, label=f"Median ({median_class})")

plt.axvline(modus_pos, linestyle=":", linewidth=2, label=f"Modus ({modus_class})")

expanded = np.repeat(x_mid, f)

kde = gaussian_kde(expanded)

x_vals = np.linspace(2.6, 3.8, 300)

kde_vals = kde(x_vals)

plt.plot(

    np.interp(x_vals, x_mid, np.arange(len(x_mid))),

    kde_vals * np.sum(f),

    linewidth=2,

    label="KDE"

)

plt.xlabel("Rentang IPK")

plt.ylabel("Jumlah Mahasiswa")

plt.title("Histogram IPK Mahasiswa\n(Mean, Median, Modus – Data Berkelompok)")

plt.legend()

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.show()

print("\n=== STATISTIK IPK (DATA BERKELOMPOK) ===")

print(f"Mean   : {mean_ipk:.2f}")

print(f"Median : {median_class} (~ {median_ipk})")

print(f"Modus  : {modus_class} (~ {modus_ipk})")



