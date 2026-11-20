import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Atur gaya visualisasi
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("=== 1. MEMBACA DATA MENTAH ===")
# a. Baca data cuaca
print("Membaca data cuaca...")
# Pastikan nama file ini sesuai dengan yang Anda download dari script API
df_weather = pd.read_csv('data/raw/data_cuaca_gabungan.csv') 
df_weather['date'] = pd.to_datetime(df_weather['date']).dt.date
df_weather['lat_round'] = df_weather['latitude'].round(1)
df_weather['lon_round'] = df_weather['longitude'].round(1)

# b. Baca data hotspot VIIRS 
# (Asumsi Anda punya file gabungan viirs-jpss1_2018_2025_Indonesia.csv di folder data/raw)
print("Membaca data titik api VIIRS...")
df_viirs = pd.read_csv('data/raw/viirs-jpss1_2018_2025_Indonesia.csv')
df_viirs['acq_date'] = pd.to_datetime(df_viirs['acq_date']).dt.date
df_viirs['lat_round'] = df_viirs['latitude'].round(1)
df_viirs['lon_round'] = df_viirs['longitude'].round(1)

print("\n=== 2. PROSES PENGGABUNGAN DATA (MERGING) ===")
# Hitung jumlah hotspot per grid per hari
viirs_daily_grid = df_viirs.groupby(['acq_date', 'lat_round', 'lon_round']).size().reset_index(name='hotspot_count')

# Gabungkan cuaca dan jumlah hotspot
df = pd.merge(
    df_weather, 
    viirs_daily_grid, 
    left_on=['date', 'lat_round', 'lon_round'], 
    right_on=['acq_date', 'lat_round', 'lon_round'], 
    how='left'
)

# Imputasi Missing Values: Jika tidak ada data hotspot, berarti aman (0)
df['hotspot_count'] = df['hotspot_count'].fillna(0)
# Buat label biner
df['is_hotspot'] = (df['hotspot_count'] > 0).astype(int)

# Buang kolom yang tidak perlu lagi
df = df.drop(columns=['lat_round', 'lon_round', 'acq_date', 'hotspot_count'])

print("\n=== 3. INFORMASI DATASET GABUNGAN ===")
print(df.info())
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Simpan data gabungan ini agar nanti bisa langsung dipakai oleh Notebook 02
import os

print("\n=== 3. INFORMASI DATASET GABUNGAN ===")
print(df.info())
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== 4. MENYIMPAN DATA ===")
# Deteksi lokasi folder saat ini
current_dir = os.getcwd()
print(f"Lokasi Notebook saat ini berjalan: {current_dir}")

# Menentukan letak folder data yang benar
if os.path.exists('data/raw'):
    # Jika notebook dijalankan di folder utama (Prediksi-Kobong)
    base_dir = '.' 
elif os.path.exists('../data/raw'):
    # Jika notebook dijalankan di dalam folder (notebooks)
    base_dir = '..'
else:
    print("WARNING: Folder data/raw tidak ditemukan. Pastikan Anda sudah membuat folder data/raw!")
    base_dir = None

# Membuat folder dan menyimpan file di tempat yang tepat
if base_dir is not None:
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    
    # Buat folder processed jika belum ada
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    # Simpan CSV
    output_path = os.path.join(processed_dir, 'dataset_gabungan.csv')
    df.to_csv(output_path, index=False)
    
    # Tampilkan lokasi pasti filenya agar mudah dicari
    print(f"\n[SUKSES] Data gabungan telah dibuat dan disimpan tepat di:\n{os.path.abspath(output_path)}")