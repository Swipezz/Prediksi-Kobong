
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# 0. KONFIGURASI
# ==========================================

sns.set_theme(style="whitegrid")

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

# Lokasi root proyek
BASE_DIR = Path(__file__).resolve().parent.parent

# Lokasi dataset
DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_gabungan.csv"

# Folder penyimpanan hasil EDA
OUTPUT_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 55)

# ==========================================
# 1. MEMBACA DATASET
# ==========================================

print("\n=== 1. MEMBACA DATASET ===")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH}\n"
        "Jalankan src/cek_data.py terlebih dahulu."
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset berhasil dimuat!")
print(f"Lokasi dataset : {DATA_PATH}")
print(f"Jumlah baris   : {len(df):,}")
print(f"Jumlah kolom   : {len(df.columns)}")

if df.empty:
    raise ValueError("Dataset kosong.")

if 'is_hotspot' not in df.columns:
    raise ValueError("Kolom 'is_hotspot' tidak ditemukan.")

# ==========================================
# EDA 1: DISTRIBUSI KELAS TARGET
# ==========================================

print("\n=== EDA 1: DISTRIBUSI KELAS TARGET ===")

df_target = df.dropna(subset=['is_hotspot']).copy()

fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(
    data=df_target,
    x='is_hotspot',
    hue='is_hotspot',
    order=[0, 1],
    palette={
        0: '#1f77b4',
        1: '#ff7f0e'
    },
    legend=False,
    ax=ax
)

ax.set_title(
    'Distribusi Kelas (0: Aman, 1: Ada Hotspot)',
    fontsize=14,
    fontweight='bold'
)

ax.set_xlabel('is_hotspot', fontsize=12)
ax.set_ylabel('Jumlah Data', fontsize=12)
ax.set_xticks([0, 1], ['0 (Aman)', '1 (Ada Hotspot)'])

# Anotasi jumlah dan persentase
total = len(df_target)

for p in ax.patches:
    height = p.get_height()

    if height > 0:
        percentage = (height / total) * 100

        ax.annotate(
            f'{int(height):,}\n({percentage:.2f}%)',
            (
                p.get_x() + p.get_width() / 2,
                height
            ),
            ha='center',
            va='bottom',
            fontsize=11
        )

plt.tight_layout()

# Simpan grafik
output_file = OUTPUT_DIR / "eda_1_distribusi_kelas.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Grafik disimpan: {output_file}")

# Ringkasan distribusi kelas
class_counts = df_target['is_hotspot'].value_counts().sort_index()

print("\nDistribusi Kelas:")

for label, count in class_counts.items():
    category = 'Aman' if label == 0 else 'Ada Hotspot'
    percentage = (count / total) * 100

    print(f"{category}: {count:,} data ({percentage:.2f}%)")

if len(class_counts) == 2 and class_counts.min() > 0:
    imbalance_ratio = class_counts.max() / class_counts.min()

    print(f"Rasio ketidakseimbangan: {imbalance_ratio:.2f}:1")

    if imbalance_ratio >= 2:
        print("Insight: Dataset mengalami class imbalance.")
    else:
        print("Insight: Distribusi kelas relatif seimbang.")

# ==========================================
# EDA 2: BOXPLOT SUHU MAKSIMUM
# ==========================================

print("\n=== EDA 2: BOXPLOT SUHU MAKSIMUM ===")

temperature_column = 'temperature_2m_max'

if temperature_column not in df.columns:
    print(
        f"Kolom '{temperature_column}' tidak ditemukan. "
        "EDA 2 dilewati."
    )
else:
    df_box = df.dropna(
        subset=['is_hotspot', temperature_column]
    ).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df_box,
        x='is_hotspot',
        y=temperature_column,
        hue='is_hotspot',
        order=[0, 1],
        palette={
            0: '#1f77b4',
            1: '#ff7f0e'
        },
        legend=False,
        ax=ax
    )

    ax.set_title(
        'Distribusi Suhu Maksimum terhadap Kejadian Hotspot',
        fontsize=14,
        fontweight='bold'
    )

    ax.set_xlabel(
        'Kejadian Hotspot (0 = Aman, 1 = Kebakaran)',
        fontsize=12
    )

    ax.set_ylabel('Suhu Maksimum (°C)', fontsize=12)

    ax.set_xticks(
        [0, 1],
        ['0 (Aman)', '1 (Hotspot)']
    )

    plt.tight_layout()

    # Simpan grafik
    output_file = OUTPUT_DIR / "eda_2_boxplot_suhu.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Grafik disimpan: {output_file}")

    # Ringkasan median suhu
    print("\nMedian Suhu Maksimum:")

    median_temperature = (
        df_box.groupby('is_hotspot')[temperature_column]
        .median()
    )

    print(median_temperature.to_string())

# ==========================================
# EDA 3: HEATMAP KORELASI
# ==========================================

print("\n=== EDA 3: HEATMAP KORELASI ===")

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

if len(numeric_cols) < 2:
    print("Kolom numerik tidak cukup untuk menghitung korelasi.")
else:
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=0.5,
        center=0,
        ax=ax
    )

    ax.set_title(
        'Korelasi antar Fitur Cuaca dan Kejadian Hotspot',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    # Simpan grafik
    output_file = OUTPUT_DIR / "eda_3_heatmap_korelasi.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Grafik disimpan: {output_file}")

    # Korelasi fitur terhadap target
    if 'is_hotspot' in corr.columns:

        target_corr = (
            corr['is_hotspot']
            .drop('is_hotspot')
            .sort_values(ascending=False)
        )

        print("\nKorelasi Fitur terhadap Target:")
        print(target_corr.to_string())

        if not target_corr.empty:
            strongest = target_corr.abs().idxmax()
            value = target_corr[strongest]

            print(
                f"\nKorelasi terkuat berdasarkan nilai absolut: "
                f"{strongest} ({value:.3f})"
            )

# ==========================================
# 4. RINGKASAN HASIL EDA
# ==========================================

print("\n" + "=" * 55)
print("EDA SELESAI")
print("=" * 55)

print(f"Total data dianalisis : {len(df):,}")
print(f"Total fitur           : {len(df.columns)}")
print(f"Folder output         : {OUTPUT_DIR}")

print("\nFile PNG yang dihasilkan:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"- {file.name}")

print("\nSemua grafik berhasil diproses.")
