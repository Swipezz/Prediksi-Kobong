import os
import time
import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry

# 1. Setup Open-Meteo API client dengan cache dan retry
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"
nama_file_output = "data_cuaca_gabungan.csv"

# Fungsi untuk membuat grid koordinat dari bounding box
def get_grid_coords(bbox_str, step=2.0):
    min_lat, min_lon, max_lat, max_lon = map(float, bbox_str.split(','))
    lats = np.arange(min_lat, max_lat + 0.01, step)
    lons = np.arange(min_lon, max_lon + 0.01, step)
    
    lat_list, lon_list = [], []
    for lat in lats:
        for lon in lons:
            lat_list.append(round(lat, 4))
            lon_list.append(round(lon, 4))
    return lat_list, lon_list

# 2. Cek data yang sudah pernah diunduh sebelumnya (Fitur Resume)
titik_selesai = set()
if os.path.exists(nama_file_output):
    try:
        df_existing = pd.read_csv(nama_file_output, usecols=['latitude', 'longitude'])
        for _, row in df_existing.drop_duplicates().iterrows():
            titik_selesai.add((round(row['latitude'], 4), round(row['longitude'], 4)))
        print(f"RESUME: Ditemukan {len(titik_selesai)} titik yang sudah berhasil diunduh di file CSV.")
    except Exception as e:
        print(f"Gagal membaca file CSV yang ada: {e}. Akan mulai dari awal.")

# 3. Definisikan area bounding box
daftar_area = [
    {"nama": "Area 1 (Barat/Sumatra)", "bounding_box": "-6.5,95,6.5,106.5"},
    {"nama": "Area 2 (Timur/Kalimantan)", "bounding_box": "-5,107.5,7,120"}
]

# Kumpulkan semua target titik dari kedua area
target_lats = []
target_lons = []

for area in daftar_area:
    lats, lons = get_grid_coords(area["bounding_box"], step=2.0)
    for lat, lon in zip(lats, lons):
        if (lat, lon) not in titik_selesai:
            target_lats.append(lat)
            target_lons.append(lon)

total_target = len(target_lats)
print(f"Total titik yang MASIH HARUS diunduh: {total_target} titik\n")

if total_target == 0:
    print("Semua data sudah terunduh lengkap!")
    exit()

# 4. Proses Request ke API
chunk_size = 2 
titik_terproses = 0 # Variabel untuk menghitung progres

for i in range(0, total_target, chunk_size):
    lat_chunk = target_lats[i:i + chunk_size]
    lon_chunk = target_lons[i:i + chunk_size]
    
    params = {
        "latitude": lat_chunk,
        "longitude": lon_chunk,
        "start_date": "2019-01-01",
        "end_date": "2026-11-10",
        "daily": [
            "precipitation_sum", 
            "temperature_2m_max", 
            "et0_fao_evapotranspiration", 
            "relative_humidity_2m_min", 
            "wind_speed_10m_max", 
            "vapour_pressure_deficit_max", 
            "soil_moisture_0_to_7cm_mean", 
            "soil_moisture_7_to_28cm_mean"
        ],
        "timezone": "GMT",
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        
        batch_data = []
        
        for idx, response in enumerate(responses):
            req_lat = lat_chunk[idx]
            req_lon = lon_chunk[idx]
            
            # Hitung persentase progres
            titik_terproses += 1
            persentase = (titik_terproses / total_target) * 100
            
            # Tampilkan informasi progres realtime
            print(f"[{titik_terproses}/{total_target}] ({persentase:.1f}%) -> Berhasil ditarik: Lintang {req_lat}°, Bujur {req_lon}°")
            
            daily = response.Daily()
            
            daily_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                )
            }
            
            daily_data["precipitation_sum"] = daily.Variables(0).ValuesAsNumpy()
            daily_data["temperature_2m_max"] = daily.Variables(1).ValuesAsNumpy()
            daily_data["et0_fao_evapotranspiration"] = daily.Variables(2).ValuesAsNumpy()
            daily_data["relative_humidity_2m_min"] = daily.Variables(3).ValuesAsNumpy()
            daily_data["wind_speed_10m_max"] = daily.Variables(4).ValuesAsNumpy()
            daily_data["vapour_pressure_deficit_max"] = daily.Variables(5).ValuesAsNumpy()
            daily_data["soil_moisture_0_to_7cm_mean"] = daily.Variables(6).ValuesAsNumpy()
            daily_data["soil_moisture_7_to_28cm_mean"] = daily.Variables(7).ValuesAsNumpy()
            
            daily_dataframe = pd.DataFrame(data=daily_data)
            daily_dataframe.insert(0, 'longitude', req_lon)
            daily_dataframe.insert(0, 'latitude', req_lat)
            
            batch_data.append(daily_dataframe)

        # Simpan langsung ke CSV (Append)
        if batch_data:
            df_chunk = pd.concat(batch_data, ignore_index=True)
            tulis_header = not os.path.exists(nama_file_output)
            df_chunk.to_csv(nama_file_output, mode='a', header=tulis_header, index=False)
            
        time.sleep(1) # Jeda 1 detik
        
    except Exception as e:
        print(f" [!] ERROR pada batch Lintang {lat_chunk}, Bujur {lon_chunk}: {e}")
        print("     Menunggu 5 detik sebelum melanjutkan ke batch berikutnya...")
        time.sleep(5)

print("\nSelesai! Semua data yang ditargetkan sudah berada di dalam file", nama_file_output)