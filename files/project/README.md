# Prediksi Hotspot (Titik Api) H+1 — Sumatra & Kalimantan

Proyek ML mengikuti lifecycle **L–M–O–P**. Repositori ini berisi hasil **Minggu 2 (tahap L: Data Understanding & Data Preparation)**.

**Target:** `fire_next_day` — ada ≥1 hotspot VIIRS (kebakaran vegetasi, confidence ≠ low) dalam radius 25 km dari titik cuaca pada hari berikutnya. Klasifikasi biner, 6 titik studi, 2019–2024.
**SDG:** 13 – Penanganan Perubahan Iklim (asumsi; sesuaikan dengan Bab 2.2 laporan).

## Struktur
```
data/raw/            # LETAKKAN DI SINI (tidak di-commit): data_cuaca_semua_lokasi.csv, viirs-jpss1_2018_2025_Indonesia.csv
data/processed/      # train/val/test (*_model.csv terskala, *_engineered.csv mentah), feature_lists.json, class_weights.json, test_checksums.json
notebooks/           # 01_learning_problem_eda.ipynb, 02_data_preparation.ipynb
src/prep.py          # seluruh fungsi pipeline (label, fitur, split) — dipakai kedua notebook
models/              # preprocessor.joblib (di-fit pada train)
reports/figures/     # fig01..fig07
EAS_Laporan_Proyek_ML_LMOP.md   # laporan (Bab 4–5 terisi)
```

## Cara menjalankan ulang
```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_learning_problem_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_data_preparation.ipynb
```
Seluruh proses ± 1–2 menit; `random_state = 42`.

## Keputusan desain utama
- **Split kronologis** (bukan acak): train target 2019-01-31…2022-12-31 (8.586), val 2023 (2.148), test 2024 (2.178).
- **Test set dikunci** hingga Minggu 6; SHA-256 di `data/processed/test_checksums.json`.
- Scaler/encoder/seleksi fitur/class weight **hanya di-fit pada train**.
- Fitur riwayat hotspot (`hs_*`) dipisah sebagai grup untuk ablation di Minggu 3.
