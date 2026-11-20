
import pandas as pd
import numpy as np
import warnings
import pickle

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# =====================================================================
# 0. KONFIGURASI
# =====================================================================

print("=" * 65)
print("MEMULAI DATA PREPARATION")
print("=" * 65)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_gabungan.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# =====================================================================
# 1. LOAD DATASET GABUNGAN
# =====================================================================

print("\n1. Membaca dataset gabungan...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH}\n"
        "Jalankan src/cek_data.py terlebih dahulu."
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset berhasil dimuat: {DATA_PATH}")
print(f"Jumlah baris  : {len(df):,}")
print(f"Jumlah kolom  : {len(df.columns)}")

# Validasi dataset
if df.empty:
    raise ValueError("Dataset kosong.")

required_columns = [
    'latitude',
    'longitude',
    'date',
    'precipitation_sum',
    'temperature_2m_max',
    'et0_fao_evapotranspiration',
    'relative_humidity_2m_min',
    'wind_speed_10m_max',
    'vapour_pressure_deficit_max',
    'soil_moisture_0_to_7cm_mean',
    'soil_moisture_7_to_28cm_mean',
    'is_hotspot'
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Kolom berikut tidak ditemukan: {missing_columns}"
    )

# =====================================================================
# 2. DATA CLEANING & FEATURE ENGINEERING
# =====================================================================

print("\n2. Melakukan Data Cleaning dan Feature Engineering...")

# Konversi tanggal
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Pastikan kolom numerik memiliki tipe data yang benar
numeric_cols = [
    col for col in required_columns
    if col != 'date'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Menghapus baris tanpa tanggal atau target
df = df.dropna(subset=['date', 'is_hotspot'])

# Memastikan target hanya berisi 0 dan 1
df = df[df['is_hotspot'].isin([0, 1])].copy()

df['is_hotspot'] = df['is_hotspot'].astype(int)

# Menghapus duplikasi data berdasarkan lokasi dan tanggal
df = df.drop_duplicates(
    subset=['latitude', 'longitude', 'date']
)

# Mengurutkan berdasarkan waktu
df = df.sort_values(
    by=['date', 'latitude', 'longitude']
).reset_index(drop=True)

# Memisahkan fitur dan target
target_col = 'is_hotspot'

feature_cols = [
    'latitude',
    'longitude',
    'precipitation_sum',
    'temperature_2m_max',
    'et0_fao_evapotranspiration',
    'relative_humidity_2m_min',
    'wind_speed_10m_max',
    'vapour_pressure_deficit_max',
    'soil_moisture_0_to_7cm_mean',
    'soil_moisture_7_to_28cm_mean'
]

# Menangani missing values pada fitur
print("\nMissing values sebelum imputasi:")
print(df[feature_cols].isnull().sum())

# Imputasi menggunakan median dari keseluruhan data
# Catatan: akan diperbaiki pada tahap split agar tidak terjadi leakage.
# Di sini hanya validasi dan penghapusan baris yang tidak lengkap.
df = df.dropna(subset=feature_cols).copy()

print(f"\nJumlah data setelah cleaning: {len(df):,}")

# Definisikan X dan y
X = df[feature_cols].copy()
y = df[target_col].copy()

print("\nDistribusi target sebelum splitting:")
print(y.value_counts().sort_index())

# =====================================================================
# 3. TIME-BASED DATA SPLITTING
# =====================================================================

print("\n3. Membagi data secara kronologis (70% Train, 15% Val, 15% Test)...")

n_total = len(df)

train_size = int(n_total * 0.70)
val_size = int(n_total * 0.15)

# Train
X_train = X.iloc[:train_size].reset_index(drop=True)
y_train = y.iloc[:train_size].reset_index(drop=True)

# Validation
X_val = X.iloc[
    train_size:train_size + val_size
].reset_index(drop=True)

y_val = y.iloc[
    train_size:train_size + val_size
].reset_index(drop=True)

# Test
X_test = X.iloc[
    train_size + val_size:
].reset_index(drop=True)

y_test = y.iloc[
    train_size + val_size:
].reset_index(drop=True)

print(f"Train set : {len(X_train):,} baris")
print(f"Val set   : {len(X_val):,} baris")
print(f"Test set  : {len(X_test):,} baris")

print("\nRentang tanggal setiap dataset:")

print(
    f"Train: {df.iloc[0]['date'].date()} "
    f"sampai {df.iloc[train_size - 1]['date'].date()}"
)

print(
    f"Val  : {df.iloc[train_size]['date'].date()} "
    f"sampai {df.iloc[train_size + val_size - 1]['date'].date()}"
)

print(
    f"Test : {df.iloc[train_size + val_size]['date'].date()} "
    f"sampai {df.iloc[-1]['date'].date()}"
)

# =====================================================================
# 4. MISSING VALUE IMPUTATION
# =====================================================================

print("\n4. Menangani Missing Values...")

# Fit imputation hanya menggunakan training data
train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_val = X_val.fillna(train_medians)
X_test = X_test.fillna(train_medians)

print("Missing values berhasil ditangani.")

# =====================================================================
# 5. DATA TRANSFORMATION (SCALING)
# =====================================================================

print("\n5. Melakukan Standardisasi Data...")

scaler = StandardScaler()

# Fit hanya pada training data
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_cols
)

# Transform validation dan testing
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=feature_cols
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_cols
)

print("Standardisasi berhasil dilakukan.")

# =====================================================================
# 6. DATA BALANCING (SMOTE)
# =====================================================================

print("\n6. Menyeimbangkan kelas menggunakan SMOTE...")

print("\nDistribusi kelas sebelum SMOTE:")
print(y_train.value_counts().sort_index())

# Validasi jumlah kelas
if y_train.nunique() < 2:
    raise ValueError(
        "Data training hanya memiliki satu kelas. "
        "SMOTE membutuhkan minimal dua kelas."
    )

# Menentukan jumlah tetangga berdasarkan jumlah sampel kelas minoritas
minority_count = y_train.value_counts().min()

if minority_count < 2:
    raise ValueError(
        "Jumlah sampel kelas minoritas kurang dari 2. "
        "SMOTE tidak dapat diterapkan."
    )

k_neighbors = min(5, minority_count - 1)

smote = SMOTE(
    random_state=RANDOM_STATE,
    k_neighbors=k_neighbors
)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_scaled,
    y_train
)

# Memastikan hasil tetap DataFrame dan Series
X_train_balanced = pd.DataFrame(
    X_train_balanced,
    columns=feature_cols
)

y_train_balanced = pd.Series(
    y_train_balanced,
    name=target_col
)

print("\nDistribusi kelas setelah SMOTE:")
print(y_train_balanced.value_counts().sort_index())

# =====================================================================
# 7. MENYIMPAN DATASET HASIL PREPARATION
# =====================================================================

print("\n7. Menyimpan hasil Data Preparation...")

# Training
X_train_balanced.to_csv(
    OUTPUT_DIR / 'X_train_ready.csv',
    index=False
)

y_train_balanced.to_csv(
    OUTPUT_DIR / 'y_train_ready.csv',
    index=False
)

# Validation
X_val_scaled.to_csv(
    OUTPUT_DIR / 'X_val_ready.csv',
    index=False
)

y_val.to_csv(
    OUTPUT_DIR / 'y_val_ready.csv',
    index=False
)

# Testing
X_test_scaled.to_csv(
    OUTPUT_DIR / 'X_test_ready.csv',
    index=False
)

y_test.to_csv(
    OUTPUT_DIR / 'y_test_ready.csv',
    index=False
)

# Simpan scaler
with open(OUTPUT_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Simpan median untuk imputasi inference
with open(OUTPUT_DIR / 'imputer_medians.pkl', 'wb') as f:
    pickle.dump(train_medians, f)

# Simpan daftar fitur
with open(OUTPUT_DIR / 'feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

# =====================================================================
# 8. RINGKASAN HASIL DATA PREPARATION
# =====================================================================

print("\n" + "=" * 65)
print("DATA PREPARATION SELESAI")
print("=" * 65)

print("\nRingkasan Dataset:")
print(f"Total data awal       : {n_total:,}")
print(f"Jumlah fitur          : {len(feature_cols)}")
print(f"Training sebelum SMOTE: {len(X_train):,}")
print(f"Training setelah SMOTE: {len(X_train_balanced):,}")
print(f"Validation            : {len(X_val):,}")
print(f"Testing               : {len(X_test):,}")

print("\nFile yang berhasil disimpan:")

for file in sorted(OUTPUT_DIR.glob('*_ready.csv')):
    print(f"- {file.name}")

print("- scaler.pkl")
print("- imputer_medians.pkl")
print("- feature_columns.pkl")

print(f"\nLokasi output: {OUTPUT_DIR}")
print("\nData siap digunakan untuk tahap Modeling!")
