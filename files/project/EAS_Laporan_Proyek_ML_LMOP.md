---
## INFORMASI PROYEK

**Judul Proyek:**  
[(Contoh: "Klasifikasi Penyakit Daun Menggunakan CNN", "Prediksi Harga Rumah dengan Machine Learning", "Analisis Sentimen Ulasan Produk")]

**Nama Mahasiswa:** [Nama Lengkap]  
**NIM:** [Nomor Induk Mahasiswa]  
**Program Studi:** [Teknologi Informasi / Rekayasa Perangkat Lunak]  
**Mata Kuliah:** [Nama Mata Kuliah]  
**Dosen Pengampu:** [Nama Dosen]  
**Tahun Akademik:** [Tahun/Semester]  
**Link GitHub Repository:** [URL Repository]  
**Link Video Pembahasan:** [URL Video (YouTube/Drive)]  
**SDG yang Dipilih:** [Nomor & nama Goal, misal: SDG 3 – Kehidupan Sehat dan Sejahtera]  
**Target SDG yang Relevan:** [misal: Target 3.4 – mengurangi kematian dini akibat penyakit tidak menular]

---

## 0. KERANGKA ML LIFECYCLE (L-M-O-P)

Laporan ini disusun mengikuti **ML Lifecycle** yang terdiri dari empat tahap berulang (iteratif):

| Tahap | Singkatan | Inti Pertanyaan | Bagian Laporan |
|-------|-----------|-----------------|----------------|
| **Learning Problem** | **L** | Apa target, objective, dan datanya? | Bab 2, 3, 4, 5 |
| **Model Design** | **M** | Model family/arsitektur apa, hypothesis space, dan inductive bias/asumsi apa? | Bab 6 (6.x.1 – 6.x.3) |
| **Optimization** | **O** | Bagaimana model dilatih: loss, optimizer, tuning, regularisasi, konvergensi? | Bab 6 (6.x.4 – 6.x.6) |
| **Predict & Evaluate** | **P** | Seberapa baik prediksi model berdasarkan metrik evaluasi? | Bab 7 |

```
   L (Learning Problem) ──► M (Model Design) ──► O (Optimization) ──► P (Predict & Evaluate)
          ▲                                                                   │
          └─────────────── iterasi / perbaikan berdasarkan hasil evaluasi ────┘
```

**Catatan untuk mahasiswa:** Setiap bagian diberi label **[L]**, **[M]**, **[O]**, atau **[P]**. Pastikan Anda dapat menjelaskan keterkaitan antar tahap, misalnya bagaimana hasil **P** mendorong perbaikan pada **L/M/O** (Bab 7.5).

---

## 1. LEARNING OUTCOMES
Pada proyek ini, mahasiswa diharapkan dapat:
1. **[L]** Mengidentifikasi masalah nyata dari **ide sendiri** yang selaras dengan salah satu tujuan **SDGs** ([Agenda 2030 – sdgs.un.org/2030agenda](https://sdgs.un.org/2030agenda)), lalu merumuskan problem statement (target, objective, data) secara jelas
2. **[L]** Melakukan analisis dan eksplorasi data (EDA) secara komprehensif (**OPSIONAL**)
3. **[L]** Melakukan data preparation yang sesuai dengan karakteristik dataset
4. **[M]** Merancang tiga model machine learning dengan justifikasi hypothesis space dan inductive bias (**WAJIB**):
   - Model baseline
   - Model machine learning / advanced
   - Model deep learning (**WAJIB**)
5. **[O]** Mengoptimasi model secara sistematis (loss function, optimizer, hyperparameter tuning, regularisasi)
6. **[P]** Menggunakan metrik evaluasi yang relevan dengan jenis tugas ML dan menganalisis hasilnya
7. Melaporkan hasil eksperimen secara ilmiah dan sistematis
8. Mengunggah seluruh kode proyek ke GitHub (**WAJIB**)
9. Menerapkan prinsip software engineering dalam pengembangan proyek

---

## 2. PROJECT OVERVIEW  **[L]**

### 2.1 Latar Belakang
**Isi bagian ini dengan:**
- Mengapa proyek ini penting?
- Permasalahan umum pada domain terkait (misal: kesehatan, pendidikan, keuangan, pertanian, NLP, computer vision, dll.)
- Manfaat proyek untuk pengguna, bisnis, atau penelitian
- Studi literatur atau referensi ilmiah (minimal 1–2 sumber wajib)
- Keterkaitan masalah dengan SDGs yang dipilih (dijabarkan di Bab 2.2)

**Contoh referensi (berformat APA/IEEE):**
> Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

**[Jelaskan konteks dan latar belakang proyek]**

### 2.2 Keterkaitan dengan SDGs (WAJIB)

Ide proyek **berasal dari mahasiswa** dan harus selaras dengan **salah satu** dari 17 Sustainable Development Goals (daftar dan contoh inspirasi ada di Lampiran A).

| Komponen | Pertanyaan Panduan | Jawaban Anda |
|----------|--------------------|--------------|
| **SDG yang dipilih** | Goal nomor berapa dan apa namanya? | [ ... ] |
| **Target SDG** | Target (misal 3.4, 13.1) mana yang paling relevan? | [ ... ] |
| **Masalah nyata** | Masalah konkret apa yang diangkat? Siapa yang terdampak? | [ ... ] |
| **Peran ML** | Mengapa ML/DL cocok? Apa keputusan atau proses yang dibantu? | [ ... ] |
| **Indikator kontribusi** | Bagaimana hasil model berkontribusi pada target SDG (langsung/tidak langsung)? | [ ... ] |
| **Pertimbangan etika** | Risiko bias, privasi, atau dampak salah prediksi? | [ ... ] |

**Sumber ide:** [Jelaskan dari mana ide muncul: pengalaman pribadi, lingkungan sekitar, isu lokal/nasional, dll.]

### 2.3 Ringkasan Alur Lifecycle Proyek
Jelaskan dalam 1 paragraf pendek bagaimana proyek Anda melewati tahap L → M → O → P (dan iterasi apa yang direncanakan).

**[Tulis ringkasan alur lifecycle proyek Anda]**

---

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING  **[L]**

### 3.1 Problem Statements
Tuliskan 2–4 pernyataan masalah yang jelas dan spesifik.

**Contoh (universal):**
1. Model perlu mampu memprediksi nilai target dengan akurasi tinggi
2. Sistem harus dapat mengidentifikasi pola pada citra secara otomatis
3. Dataset memiliki noise sehingga perlu preprocessing yang tepat
4. Dibutuhkan model deep learning yang mampu belajar representasi fitur kompleks

**[Tulis problem statements Anda di sini]**

### 3.2 Formulasi Learning Problem (WAJIB)

Rumuskan masalah dalam kerangka *Learning Problem*:

| Komponen | Pertanyaan Panduan | Jawaban Anda |
|----------|--------------------|--------------|
| **Target** | Apa yang diprediksi? Jenis tugas (klasifikasi biner/multikelas, regresi, forecasting, clustering, dll.)? | [ ... ] |
| **Objective** | Apa yang dioptimalkan/diminimalkan secara matematis (misal: cross-entropy, MSE)? Metrik keberhasilan utama? | [ ... ] |
| **Data** | Sumber, ukuran, jenis, dan asumsi tentang data (i.i.d.? distribusi train vs test?) | [ ... ] |

### 3.3 Goals

Tujuan harus spesifik, terukur, dan selaras dengan problem statement.
**Contoh tujuan:**
1. Membangun model ML untuk memprediksi variabel target dengan akurasi > 80%
2. Mengukur performa tiga pendekatan model (baseline, advanced, deep learning)
3. Menentukan model terbaik berdasarkan metrik evaluasi yang relevan
4. Menghasilkan sistem yang dapat bekerja secara reproducible

> Sertakan minimal satu goal yang menunjukkan kontribusi terhadap target SDG yang dipilih (Bab 2.2).

**[Tulis goals Anda di sini]**

### 3.4 Solution Approach  **[M]**

Mahasiswa **WAJIB** menggunakan minimal **tiga model** dengan komposisi sebagai berikut:
#### **Model 1 – Baseline Model**
Model sederhana sebagai pembanding dasar.
**Pilihan model:**
- Linear Regression (untuk regresi)
- Logistic Regression (untuk klasifikasi)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes

**[Jelaskan model baseline yang Anda pilih dan alasannya]**

#### **Model 2 – Advanced / ML Model**
Model machine learning yang lebih kompleks.
**Pilihan model:**
- Random Forest
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Support Vector Machine (SVM)
- Ensemble methods
- Clustering (K-Means, DBSCAN) - untuk unsupervised
- PCA / dimensionality reduction (untuk preprocessing)

**[Jelaskan model advanced yang Anda pilih dan alasannya]**

#### **Model 3 – Deep Learning Model (WAJIB)**
Model deep learning yang sesuai dengan jenis data.
**Pilihan Implementasi (pilih salah satu sesuai dataset):**
**A. Tabular Data:**
- Multilayer Perceptron (MLP) / Neural Network
- Minimum: 2 hidden layers
- Contoh: prediksi harga, klasifikasi binary/multiclass

**B. Image Data:**
- CNN sederhana (minimum 2 convolutional layers) **ATAU**
- Transfer Learning (ResNet, VGG, MobileNet, EfficientNet) - **recommended**
- Contoh: klasifikasi gambar, object detection

**C. Text Data:**
- LSTM/GRU (minimum 1 layer) **ATAU**
- Embedding + Dense layers **ATAU**
- Pre-trained model (BERT, DistilBERT, Word2Vec)
- Contoh: sentiment analysis, text classification

**D. Time Series:**
- LSTM/GRU untuk sequential prediction
- Contoh: forecasting, anomaly detection

**E. Recommender Systems:**
- Neural Collaborative Filtering (NCF)
- Autoencoder-based Collaborative Filtering
- Deep Matrix Factorization

**Minimum Requirements untuk Deep Learning:**
- ✅ Model harus training minimal 10 epochs
- ✅ Harus ada plot loss dan accuracy/metric per epoch
- ✅ Harus ada hasil prediksi pada test set
- ✅ Training time dicatat (untuk dokumentasi)

**Tidak Diperbolehkan:**
- ❌ Copy-paste kode tanpa pemahaman
- ❌ Model tidak di-train (hanya define arsitektur)
- ❌ Tidak ada evaluasi pada test set

**[Jelaskan model deep learning yang Anda pilih dan alasannya]**

---

## 4. DATA UNDERSTANDING  **[L]**

> **Catatan formulasi (asumsi):** Bab 3 pada template ini belum terisi, sehingga definisi target di Bab 4–5 disimpulkan dari kedua dataset yang digunakan. **Target:** `fire_next_day` = ada ≥ 1 hotspot VIIRS (kebakaran vegetasi, confidence ≠ low) dalam radius 25 km dari titik cuaca pada hari berikutnya (H+1) — klasifikasi biner. SDG yang paling sesuai: SDG 13 (Penanganan Perubahan Iklim). Mohon sesuaikan bila rumusan di Bab 3.2 berbeda.

### 4.1 Informasi Dataset
**Sumber Dataset:**  
1. **Data cuaca harian** — `data_cuaca_semua_lokasi.csv`. Variabel dan formatnya konsisten dengan keluaran Open-Meteo Historical Weather API (https://open-meteo.com/) *(sumber diduga dari struktur berkas; mohon dikonfirmasi)*.
2. **Hotspot VIIRS NOAA-20 (JPSS-1)** — `viirs-jpss1_2018_2025_Indonesia.csv`, format NASA FIRMS (https://firms.modaps.eosdis.nasa.gov/) *(mohon konfirmasi URL unduhan)*.

**Deskripsi Dataset:**

| | Cuaca harian | Hotspot VIIRS |
|---|---|---|
| Jumlah baris (rows) | 17.226 (6 lokasi × 2.871 hari) | 1.213.145 |
| Jumlah kolom (columns) | 11 | 16 |
| Tipe data | Tabular / time series spasial | Tabular geospasial (satu baris = satu deteksi piksel) |
| Ukuran dataset | 2,43 MB | 103,80 MB |
| Format file | CSV | CSV |
| Rentang waktu | 2019-01-01 s.d. 2026-11-10 | 2018-04-01 s.d. 2024-12-31 |

Data yang dimodelkan (setelah diproses pada Bab 5) berisi **12.912 baris** (lokasi × hari) dengan **28 fitur numerik awal** (22 setelah seleksi) ditambah one-hot lokasi. Periode tumpang-tindih kedua dataset dan berlabel adalah 2019-01-01 s.d. 2024-12-31. Catatan: nama file VIIRS menyebut 2025, tetapi isi data berakhir 2024-12-31.

### 4.2 Deskripsi Fitur

**(a) Data cuaca** (satu baris per lokasi per hari, hari berbasis UTC)

| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
|------------|-----------|-----------|--------------|
| latitude | Float | Lintang titik cuaca (6 titik unik, ID L1–L6) | -3.268893 |
| longitude | Float | Bujur titik cuaca | 105.266556 |
| date | String → datetime | Tanggal (UTC) | 2019-01-01 00:00:00+00:00 |
| precipitation_sum | Float | Curah hujan harian (mm) | 17.5 |
| temperature_2m_max | Float | Suhu udara maksimum pada 2 m (°C) | 29.45 |
| et0_fao_evapotranspiration | Float | Evapotranspirasi acuan FAO-56 (mm/hari) | 3.05 |
| relative_humidity_2m_min | Float | Kelembapan relatif minimum (%) | 75.65 |
| wind_speed_10m_max | Float | Kecepatan angin maksimum pada 10 m (satuan default Open-Meteo, km/jam) | 15.41 |
| vapour_pressure_deficit_max | Float | Defisit tekanan uap maksimum (kPa) | 0.997 |
| soil_moisture_0_to_7cm_mean | Float | Rata-rata kelembapan tanah 0–7 cm (m³/m³) | 0.419 |
| soil_moisture_7_to_28cm_mean | Float | Rata-rata kelembapan tanah 7–28 cm (m³/m³) | 0.387 |

**(b) Data hotspot VIIRS**

| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai | Dipakai? |
|------------|-----------|-----------|--------------|----------|
| latitude, longitude | Float | Koordinat pusat piksel deteksi (resolusi 375 m) | -8.20034, 119.0813 | Ya (jarak ke titik cuaca) |
| bright_ti4 | Float | Suhu kecerahan kanal I-4 (K) | 317.37 | Tidak |
| bright_ti5 | Float | Suhu kecerahan kanal I-5 (K) | 291.25 | Tidak |
| scan, track | Float | Ukuran piksel arah *scan* / *track* (km) | 0.42, 0.45 | Tidak |
| acq_date | String → datetime | Tanggal akuisisi (UTC) | 2018-04-01 | Ya (kunci waktu) |
| acq_time | Integer | Jam akuisisi UTC (HHMM) | 1701 | Tidak |
| satellite / instrument / version | Kategorikal | Konstan (N20 / VIIRS / 2) | N20, VIIRS, 2 | Tidak (konstan) |
| confidence | Kategorikal | Tingkat kepercayaan: n (nominal), l (low), h (high) | n | Ya (filter) |
| frp | Float | *Fire Radiative Power* (MW) | 2.87 | Ya (agregat) |
| daynight | Kategorikal | Deteksi siang (D) / malam (N) | N | Tidak |
| type | Integer | 0 kebakaran vegetasi; 1 gunung api; 2 sumber statis; 3 lepas pantai | 0 | Ya (filter) |
| source_year | Integer | Penanda berkas sumber (bukan tahun kejadian) | 2018 | Tidak |

**(c) Fitur turunan & target** dijelaskan di Bab 5.2. Target: `fire_next_day` (Integer 0/1).

### 4.3 Kondisi Data

- **Missing Values:** Tidak ada (0 sel kosong pada cuaca maupun VIIRS; 0%). Namun terdapat **7 hari tanpa satupun hotspot di seluruh Indonesia** (2023-07-19…22, 2023-09-30, 2024-02-04, 2024-02-05; median normal ≈ 159/hari) — diduga data satelit hilang, diperlakukan sebagai *missing* (Bab 5.1d).
- **Duplicate Data:** Tidak ada (0 baris duplikat penuh; 0 duplikat kunci lokasi-tanggal pada cuaca; 0 duplikat pada VIIRS).
- **Outliers:** Ada pada fitur cuaca (aturan IQR 1,5×): curah hujan 615 baris (3,6%; skewness 2,34), VPD 652 (3,8%), suhu maks 574 (3,3%), RH min 484 (2,8%), ET0 202 (1,2%), angin 137 (0,8%), kelembapan tanah 0–7 cm 31 (0,2%), 7–28 cm 0. Semua masih dalam rentang fisik yang wajar (RH 22,5–99,7%; hujan maks 155 mm/hari). FRP VIIRS sangat miring ke kanan (median 4,6 MW; persentil ke-99,9 = 170 MW; maks 1.841 MW) — kebakaran besar yang nyata.
- **Imbalanced Data:** Ada. Pada train: 858 positif vs 7.728 negatif (**10,0% positif, rasio ≈ 9:1**). Proporsi positif juga sangat berbeda antar-tahun: 32,0% (2019), 3,1% (2020), 3,7% (2021), 2,9% (2022), 20,7% (2023, validasi).
- **Noise:** VIIRS memuat 91.459 deteksi bukan kebakaran vegetasi (type 1–3; 7,5%) dan 82.965 deteksi *low confidence* pada type 0; setelah disaring tersisa 1.038.721 deteksi (85,6%).
- **Data Quality Issues:**
  1. Cakupan waktu tidak sama: VIIRS 2018-04…2024-12-31, cuaca 2019-01…2026-11-10 → cuaca setelah 2024-12-31 (4.080 baris) tidak memiliki label.
  2. Hanya **6 titik** (L1–L6; ≈ Sumatra bagian selatan dan Kalimantan); L2–L3 hanya berjarak ≈ 74 km, titik lainnya berjarak ≥ 170 km — generalisasi ke lokasi baru tidak dapat diuji.
  3. Data cuaca adalah keluaran model/reanalisis berbasis grid, bukan pengukuran stasiun.
  4. Multikolinearitas tinggi antar-fitur (mis. RH min vs VPD |ρ| = 0,99).
- **Asumsi Data (i.i.d., distribusi train-test, potensi data leakage):**
  - Data **tidak i.i.d.**: deret waktu dengan autokorelasi temporal dan spasial; hari bertetangga hampir identik.
  - Distribusi **tidak stasioner**: tahun api besar (2019, 2023) vs tahun basah (2020–2022); train (10% positif) berbeda dari validasi (20,7%). Skor validasi karenanya tidak boleh dibaca sebagai taksiran langsung performa test.
  - Sumber potensi leakage: (a) fitur rolling yang melihat masa depan → dicegah dengan jendela mundur dan **diuji** (Bab 5.2); (b) split acak pada data deret waktu → dipakai split kronologis; (c) *scaler*/seleksi fitur yang di-fit pada seluruh data → di-fit hanya pada train; (d) hari gap satelit menjadi label 0 palsu → dibuang. Catatan: fitur riwayat hotspot (`hs_*`) sah untuk skenario forecasting karena diketahui pada hari t, tetapi sangat dominan sehingga perlu *ablation* pada Minggu 3.

### 4.4 Exploratory Data Analysis (EDA) - (**OPSIONAL**)

Analisis hubungan fitur–target (Visualisasi 3–6) hanya memakai **data train** (Visualisasi 6 juga memuat validasi); **data test tidak dilihat**. Kode: `notebooks/01_learning_problem_eda.ipynb`.

#### Visualisasi 1: Hotspot bulanan 2018–2023 (musiman & variasi antar-tahun)
![Hotspot bulanan](reports/figures/fig01_hotspot_bulanan.png)

**Insight:**  
Hotspot sangat musiman: puncaknya selalu Agustus–Oktober (musim kemarau), dan nyaris nol pada Desember–Juni. Intensitas antar-tahun berbeda drastis: 2019 (≈391 ribu deteksi nasional) dan 2023 (≈215 ribu) jauh di atas 2020–2022 (≈50–77 ribu per tahun, tahun basah/La Niña). Di 6 titik studi pola yang sama muncul: kebakaran hanya besar pada 2019 dan 2023. Konsekuensi: (1) fitur kalender/musiman (`doy_sin`, `doy_cos`) layak dibuat; (2) proporsi kelas positif sangat bergantung pada tahun sehingga **ada pergeseran distribusi antar-tahun** — split acak akan menutupi masalah ini, sehingga dipakai split kronologis.

#### Visualisasi 2: Sebaran spasial hotspot dan posisi 6 titik studi

![Peta hotspot dan titik studi](reports/figures/fig02_peta_hotspot_titik.png)

**Insight:**  
Keenam titik berada di pulau Sumatra bagian selatan (L1–L3) dan Kalimantan (L4–L6) — dua kawasan dengan konsentrasi hotspot tertinggi di Indonesia pada peta (sel merah tua), sesuai dengan kawasan lahan gambut/lahan kering rawan karhutla. L2 dan L3 hanya berjarak ≈74 km sehingga buffer 25 km tidak saling tumpang tindih (label tiap titik independen). Model yang dilatih pada data ini hanya mewakili 6 lokasi tersebut; generalisasi ke lokasi lain **tidak** diuji (keterbatasan yang dicatat).

#### Visualisasi 3: Peluang hotspot besok menurut bulan dan lokasi (train)

![Heatmap bulan x lokasi](reports/figures/fig03_heatmap_bulan_lokasi.png)

**Insight:**  
Ada musim api yang jelas: rata-rata proporsi hari ber-hotspot naik dari ≈2–5% (Jan–Jun) menjadi ≈15% (Jul), 22% (Agu), 27% (Sep), 16% (Okt), 13% (Nov), lalu turun lagi di Des. Puncak Sumatra Selatan (L2, L3) bergeser ke September–Oktober (hingga 0,29–0,33), sedangkan Kalimantan (L4–L6) memuncak Agustus–September. L1 (Jambi) berbeda: api muncul sepanjang tahun (6–25%) sehingga relatif sulit dibedakan dari musimnya saja. Catatan: angka ini dihitung dari data train yang didominasi tahun 2019, sehingga level absolutnya bias ke tahun api besar.

#### Visualisasi 4: Fitur kunci menurut kelas target (train)

![Boxplot fitur vs target](reports/figures/fig04_boxplot_fitur_vs_target.png)

**Insight:**  
Hari sebelum ada hotspot (kelas 1) memiliki kelembapan tanah 7–28 cm lebih rendah (median 0,31 vs 0,42), VPD lebih tinggi (2,01 vs 1,28 kPa), hujan 30 hari jauh lebih sedikit (median 77,5 vs 260 mm) dan lebih banyak hari kering berturut-turut. Fitur riwayat hotspot paling tajam memisahkan kelas: median hotspot 7 hari terakhir ≈ 0 untuk kelas 0 tetapi ≈ 24 untuk kelas 1, dan hari sejak hotspot terakhir median 31 vs 0. Artinya prediksi sebagian besar berasal dari **persistensi** (api hari ini → api besok) ditambah sinyal kekeringan; Minggu 3 perlu ablation (dengan vs tanpa fitur riwayat) agar kontribusi cuaca sendiri terukur.

#### Visualisasi 5: Korelasi Spearman antar fitur (train)

![Korelasi Spearman](reports/figures/fig05_korelasi_spearman.png)

**Insight:**  
Korelasi tertinggi dengan target: `hs_count` (0,61), `hs_count_7d` (0,51), `hs_days_since_last` (−0,40), `precip_sum_30d` (−0,37), `water_deficit_14d` (0,36). Banyak fitur redundan: RH minimum vs VPD (|ρ| = 0,99), curah hujan 14 hari vs defisit air (0,996), hotspot 7 hari vs FRP 7 hari (0,996), dan kelembapan tanah dua lapisan (0,98). Multikolinearitas ini tidak merusak model berbasis pohon tetapi membuat koefisien regresi logistik tidak stabil dan membuang kapasitas MLP; karena itu dilakukan seleksi fitur berbasis korelasi (ambang |ρ| > 0,97) **yang dihitung hanya pada train** di notebook 02.

#### Visualisasi 6: Distribusi kelas per tahun dan per lokasi

![Kelas per tahun dan lokasi](reports/figures/fig06_kelas_per_tahun_lokasi.png)

**Insight:**  
Proporsi target=1 pada train hanya 10,0% (rasio ≈ 9:1), tetapi sangat berbeda antar tahun: 32,0% (2019), 3,1% / 3,7% / 2,9% (2020–2022), dan 20,7% pada validasi 2023. Antar lokasi 7–15% (L1 tertinggi). Jadi (a) kelas positif minoritas → metrik utama PR-AUC/F1, bukan accuracy; (b) distribusi train ≠ validasi: validasi memiliki lebih banyak kelas positif, sehingga skor validasi tidak bisa dibaca sebagai prediksi langsung performa test dan perlu kehati-hatian saat memilih threshold.

---

## 5. DATA PREPARATION  **[L]**

Bagian ini menjelaskan **semua** proses transformasi dan preprocessing data. Kode: `src/prep.py` (fungsi) dan `notebooks/02_data_preparation.ipynb` (eksekusi + verifikasi). Seluruh langkah memakai `random_state = 42`.

**Urutan eksekusi** (sedikit berbeda dari urutan bab, demi mencegah leakage): 5.1 → 5.2 (tanpa parameter yang dipelajari dari data) → **5.4 split** → 5.3 (parameter di-*fit* hanya pada train) → 5.5.

### 5.1 Data Cleaning

```
Missing Values:
- Cuaca & VIIRS: 0 missing (0%). Tidak ada imputasi.
- Alasan: imputasi pada deret waktu berisiko menyuntikkan informasi dari masa depan; verifikasi dilakukan dengan assert agar pipeline gagal-keras bila data berubah.

Hari gap satelit:
- 7 hari dengan 0 hotspot di seluruh Indonesia (median normal 159/hari) -> hitungan hotspot dijadikan NaN, bukan 0.
- Alasan: mencegah label negatif palsu. Baris yang label-nya jatuh pada hari gap (42 baris) dan baris dengan hari-t gap (18 baris) dibuang.

Duplicates:
- 0 duplikat -> tidak ada yang dihapus (diverifikasi dengan assert).

Noise VIIRS:
- Hanya type == 0 (kebakaran vegetasi) dan confidence != 'low': 1.213.145 -> 1.121.686 (92,5%) -> 1.038.721 (85,6%).
- Alasan: type 1-3 (gunung api, sumber statis, lepas pantai) bukan karhutla; confidence low banyak positif palsu.
- Kolom konstan (satellite, instrument, version) dan source_year tidak dipakai.

Outliers:
- Outlier IQR (0,2%-3,8% per fitur cuaca) TIDAK dihapus.
- Alasan: semuanya valid secara fisik (hujan lebat, hari sangat kering) dan justru informatif untuk kebakaran.
- Penanganan: log1p untuk fitur miring-kanan (skewness presipitasi 2,34 -> -0,13), lihat 5.3.

Konversi tipe & penyelarasan:
- date (cuaca, string UTC) dan acq_date (VIIRS) -> datetime tanpa zona waktu; lokasi diberi ID L1-L6.
- Hotspot diagregasi per (lokasi, hari) dalam radius 25 km (haversine): hs_count dan hs_frp_sum.
- Alasan radius 25 km: proporsi positif wajar (~10%) dan lingkaran antar-titik tidak tumpang tindih (jarak minimum ~74 km).
```

**Funnel baris (alasan setiap baris dibuang):**

| Tahap | Baris tersisa | Dibuang |
|---|---|---|
| Baris cuaca awal (6 lokasi × 2.871 hari) | 17.226 | – |
| Buang hari tanpa label (target_date > 2024-12-31; cuaca 2025–2026 tidak punya VIIRS) | 13.146 | 4.080 |
| Buang baris yang label-nya jatuh pada hari gap satelit | 13.104 | 42 |
| Buang baris dengan hari-t gap | 13.086 | 18 |
| Buang warm-up jendela rolling 30 hari (29 hari pertama per lokasi) | **12.912** | 174 |

### 5.2 Feature Engineering

Semua fitur dihitung **hanya dari hari ≤ t** (jendela mundur, per lokasi). Target = ada hotspot pada hari t+1.

| Grup | Fitur | Definisi | Alasan |
|---|---|---|---|
| Cuaca mentah (8) | 8 variabel Bab 4.2a | nilai hari t | kondisi langsung bahan bakar & atmosfer |
| Akumulasi hujan (4) | `precip_sum_{3,7,14,30}d` | jumlah hujan jendela mundur | kekeringan bersifat kumulatif |
| Spell kering (1) | `dry_days_consec` | hari berturut-turut hujan < 1 mm (maks 90) | lamanya kekeringan |
| Rata-rata 7 hari (4) | suhu, RH, VPD, angin | rolling mean 7 hari | meredam noise harian |
| Tanah (3) | `soil_0_7_mean_7d`, `soil_7_28_mean_14d`, `soil_0_7_delta_7d` | rolling mean & selisih 7 hari | tren pengeringan tanah/gambut |
| Neraca air (1) | `water_deficit_14d` | ΣET0 14 hari − Σhujan 14 hari | defisit air fisik |
| Musiman (2) | `doy_sin`, `doy_cos` | sin/cos hari-dalam-tahun | musim api siklik, kontinu di pergantian tahun |
| Riwayat hotspot (5) | `hs_count`, `hs_count_7d`, `hs_count_30d`, `hs_frp_sum_7d`, `hs_days_since_last` | hitungan/FRP/umur api terakhir (dihitung dari grid VIIRS sejak 2018-04) | persistensi api; dipisah sebagai grup agar dapat di-*ablation* |
| Lokasi | one-hot `loc_id` (L1–L6) | – | tingkat dasar tiap lokasi berbeda (7–15%) |

Total: **28 fitur numerik** + one-hot lokasi.

**Feature selection** (dihitung pada **train saja**, Bab 5.3a): 28 → **22** fitur numerik. **Dimensionality reduction (PCA) tidak dilakukan:** hanya 22 fitur dan fitur dipertahankan interpretabel untuk analisis kontribusi.

**Uji kausalitas (anti-leakage):** fitur dihitung pada data yang dipotong di 2023-06-30 dan pada data penuh; seluruh fitur pada baris ≤ tanggal itu **identik** (cuaca: lolos; riwayat hotspot: lolos), dan label terverifikasi sama dengan hotspot hari t+1.

### 5.3 Data Transformation

**Tabular (deret waktu) — semua parameter di-`fit` hanya pada train:**

1. **Seleksi fitur berbasis korelasi:** untuk tiap pasangan dengan |Spearman ρ| > 0,97 buang salah satu yang korelasinya dengan target lebih rendah. Dibuang (6): `precip_sum_14d` (↔ `water_deficit_14d`, 0,996), `hs_frp_sum_7d` (↔ `hs_count_7d`, 0,996), `vapour_pressure_deficit_max` (↔ `relative_humidity_2m_min`, 0,993), `vpd_max_mean_7d` (↔ `rh_min_mean_7d`, 0,990), `soil_moisture_0_to_7cm_mean` (↔ `soil_moisture_7_to_28cm_mean`, 0,978), `soil_0_7_mean_7d` (↔ `soil_moisture_7_to_28cm_mean`, 0,975). Pemilihan RH vs VPD secara statistik hampir setara (|ρ| dgn target 0,319 vs 0,312). Seluruh 28 fitur tetap tersimpan di `*_engineered.csv` untuk ablation.
2. **`log1p`** pada fitur berekor panjang (7 fitur terpilih): `precipitation_sum`, `precip_sum_{3,7,30}d`, `hs_count`, `hs_count_7d`, `hs_count_30d`. Deterministik (tanpa parameter) sehingga tanpa risiko leakage; menjaga nol tetap nol.
3. **One-hot encoding** `loc_id` (`handle_unknown='ignore'`) — 6 kolom.
4. **StandardScaler** pada 20 fitur numerik (di luar `doy_sin/cos`) — rata-rata & simpangan baku dari train; `doy_sin/cos` dibiarkan (sudah dalam [−1, 1]).
5. Implementasi: `ColumnTransformer.fit(train)` → `transform(val/test)`; disimpan di `models/preprocessor.joblib`. Bentuk akhir: 28 kolom model (6 one-hot + 20 terstandar + 2 siklik).

Tokenisasi, augmentasi gambar, dsb. tidak relevan (data tabular).

### 5.4 Data Splitting

**Strategi pembagian data** — kronologis berdasarkan *tanggal target* (hari yang diprediksi):

```
- Training set  : 66,5% (8.586 samples)  target 2019-01-31 s.d. 2022-12-31   | positif 858 (10,0%)
- Validation set: 16,6% (2.148 samples)  target 2023-01-01 s.d. 2023-12-31   | positif 444 (20,7%)
- Test set      : 16,9% (2.178 samples)  target 2024-01-01 s.d. 2024-12-31   | DIKUNCI (tidak dianalisis)
- Total         : 12.912 samples (tiap lokasi: 1.431 / 358 / 363)
- Random state  : 42 (diagnostik & model berikutnya)
```

**Mengapa bukan *stratified random split*?** Data adalah deret waktu spasial: hari bertetangga sangat berautokorelasi dan fitur rolling saling tumpang tindih; skenario nyata juga memprediksi masa depan dari masa lalu. Stratifikasi tidak dapat dipenuhi bersamaan dengan urutan waktu, sehingga proporsi kelas tiap split **dilaporkan** dan pembobotan kelas dihitung dari train (5.5). Pembagian 2019–2022 / 2023 / 2024 membuat train dan validasi sama-sama memuat satu tahun api besar (2019 dan 2023), sedangkan 2024 sebagai tahun terakhir menjadi test.

**Diagnostik (hanya train+val; bukan pemilihan model):** Random Forest sederhana pada 10 pengacakan stratified 80/20 menghasilkan ROC-AUC 0,930 ± 0,010 (rentang 0,915–0,943); pada split kronologis 0,929. Jadi **tidak terbukti** split acak menggelembungkan skor secara sistematis pada dataset ini; keunggulan split kronologis terletak pada kesesuaiannya dengan skenario penggunaan, estimasi yang lebih stabil, dan karena split acak menyembunyikan pergeseran distribusi (proporsi positif 12,1% vs 20,7% pada validasi 2023).

> **Penting:** Test set hanya digunakan **sekali** pada tahap P (evaluasi akhir). Tuning pada tahap O hanya boleh memakai validation set / cross-validation berbasis waktu. *Scaler*/*encoder* di-fit hanya pada train. Sidik jari SHA-256 berkas test disimpan di `data/processed/test_checksums.json` agar dapat dibuktikan tidak berubah hingga Minggu 6.

### 5.5 Data Balancing (jika diperlukan)

**Teknik yang digunakan: class weight (dihitung dari data train saja).** Validasi dan test dibiarkan pada distribusi aslinya.

```
n_train = 8.586 (negatif 7.728, positif 858)
weight_kelas_0 = n / (2 * n0) = 0,5555
weight_kelas_1 = n / (2 * n1) = 5,0035
scale_pos_weight (XGBoost/LightGBM) = n0 / n1 = 9,007
```

**Mengapa bukan SMOTE / undersampling?** (1) Baris adalah deret waktu dengan fitur rolling — sampel sintetis hasil interpolasi menghasilkan kombinasi hujan/riwayat yang tidak pernah terjadi, merusak koherensi temporal, dan membuat one-hot lokasi non-biner; (2) rasio 9:1 tergolong sedang, class weight + metrik PR-AUC/F1 memadai; (3) undersampling membuang ≈ 90% hari negatif. SMOTE dicatat sebagai kandidat iterasi di Bab 7.5 bila recall kelas positif rendah.

![Distribusi kelas per split](reports/figures/fig07_distribusi_kelas_split.png)

### 5.6 Ringkasan Data Preparation

| # | Langkah | **Apa** | **Mengapa** | **Bagaimana** |
|---|---|---|---|---|
| 1 | Penyelarasan tanggal | `date`/`acq_date` → tanggal UTC naif; lokasi diberi ID L1–L6 | kunci join harus identik | `to_datetime(utc=True)` + `normalize()` |
| 2 | Verifikasi kualitas | cek missing & duplikat | pipeline gagal-keras bila data berubah | `assert`; tanpa imputasi |
| 3 | Filter VIIRS | hanya `type==0` & confidence ≠ low (85,6% tersisa) | buang sumber non-karhutla & deteksi palsu | `prep.filter_viirs` |
| 4 | Hari gap satelit | 7 hari 0-hotspot nasional → NaN; baris terkait dibuang (60 baris) | cegah label negatif palsu | `prep.gap_days` |
| 5 | Outlier | dipertahankan | ekstrem bersifat fisik & relevan | `log1p` untuk fitur miring-kanan |
| 6 | Label & agregasi | target = ada hotspot ≤ 25 km pada t+1 | formulasi forecasting H+1 | haversine + `shift(-1)` |
| 7 | Feature engineering | 8 mentah + 13 cuaca-turunan + 5 riwayat hotspot + 2 musiman + one-hot lokasi | menangkap kekeringan kumulatif, musim, persistensi | rolling mundur per lokasi |
| 8 | Uji kausalitas | fitur hari t tak berubah bila data setelah t dibuang | cegah leakage dari jendela | potong-bandingkan data |
| 9 | Split kronologis | train 2019–22 / val 2023 / test 2024 | deret waktu & skenario forecasting | `prep.assign_split` |
| 10 | Seleksi fitur | buang pasangan \|ρ\| > 0,97 (28 → 22) | kurangi redundansi | korelasi Spearman pada train |
| 11 | Transformasi | `log1p`, StandardScaler, one-hot, sin/cos | kestabilan skala model linear/MLP | `ColumnTransformer.fit(train)` |
| 12 | Balancing | class weight dari train, tanpa resampling | hindari data sintetis tidak realistis | `n / (2·n_c)` |

---

## 6. MODELING  **[M + O]**

> Setiap model (6.1, 6.2, 6.3) dijelaskan dalam dua fase: **Model Design [M]** (sub-bagian .1–.3) dan **Optimization [O]** (sub-bagian .4–.6).

### 6.1 Model 1 — Baseline Model

#### 6.1.1 Deskripsi Model  **[M]**

**Nama Model:** [Nama model, misal: Logistic Regression]  
**Teori Singkat:**  
[Jelaskan secara singkat bagaimana model ini bekerja]  
**Alasan Pemilihan:**  
[Mengapa memilih model ini sebagai baseline?]

#### 6.1.2 Hypothesis Space & Inductive Bias  **[M]**

- **Model family / arsitektur:** [misal: model linear]
- **Hypothesis space:** [himpunan fungsi yang dapat direpresentasikan, misal: batas keputusan linear]
- **Inductive bias / asumsi:** [misal: hubungan fitur-target linear, fitur independen]
- **Implikasi terhadap dataset Anda:** [apakah asumsi cocok dengan data?]

#### 6.1.3 Implementasi (Ringkas)  **[M]**
```python
# Contoh kode (opsional, bisa dipindah ke GitHub)
from sklearn.linear_model import LogisticRegression

model_baseline = LogisticRegression(C=1.0, max_iter=100)
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)
```

#### 6.1.4 Optimization: Objective, Hyperparameter & Regularisasi  **[O]**

**Objective/loss yang dioptimalkan:** [misal: log-loss + L2]  
**Parameter yang digunakan:**
```
[Tuliskan parameter penting, contoh:]
- C (regularization): 1.0
- solver: 'lbfgs'
- max_iter: 100
```

#### 6.1.5 Hasil Awal  **[O → P]**

**[Tuliskan hasil awal pada validation set; evaluasi akhir pada test set dijelaskan di Bab 7]**

---

### 6.2 Model 2 — ML / Advanced Model

#### 6.2.1 Deskripsi Model  **[M]**

**Nama Model:** [Nama model, misal: Random Forest / XGBoost]  
**Teori Singkat:**  
[Jelaskan bagaimana algoritma ini bekerja]

**Alasan Pemilihan:**  
[Mengapa memilih model ini?]

**Keunggulan:**
- [Sebutkan keunggulan]

**Kelemahan:**
- [Sebutkan kelemahan]

#### 6.2.2 Hypothesis Space & Inductive Bias  **[M]**

- **Model family / arsitektur:** [misal: ensemble pohon keputusan]
- **Hypothesis space:** [fungsi non-linear piecewise, interaksi fitur otomatis, dll.]
- **Inductive bias / asumsi:** [misal: data tabular dengan struktur ambang/threshold]
- **Perbandingan dengan baseline:** [apa yang bisa dipelajari model ini yang tidak bisa baseline?]

#### 6.2.3 Implementasi (Ringkas)  **[M]**
```python
# Contoh kode
from sklearn.ensemble import RandomForestClassifier

model_advanced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model_advanced.fit(X_train, y_train)
y_pred_advanced = model_advanced.predict(X_test)
```

#### 6.2.4 Optimization: Hyperparameter Tuning  **[O]**

**Parameter yang digunakan:**
```
[Tuliskan parameter penting, contoh:]
- n_estimators: 100
- max_depth: 10
- learning_rate: 0.1
- min_samples_split: 2
```

**Hyperparameter Tuning (WAJIB minimal 1 metode sederhana):**
- Metode: [Grid Search / Random Search / Bayesian Optimization]
- Search space: [rentang parameter yang dicoba]
- Skema validasi: [k-fold CV / validation set]
- Best parameters: [...]
- Skor validasi sebelum vs sesudah tuning: [...]

#### 6.2.5 Kontrol Overfitting  **[O]**
[Jelaskan regularisasi/pembatasan kompleksitas yang dipakai, misal: max_depth, min_samples_leaf, early stopping pada boosting]

#### 6.2.6 Hasil Model  **[O → P]**

**[Tuliskan hasil pada validation set; evaluasi akhir dijelaskan di Bab 7]**

---

### 6.3 Model 3 — Deep Learning Model (WAJIB)

#### 6.3.1 Deskripsi Model  **[M]**

**Nama Model:** [Nama arsitektur, misal: CNN / LSTM / MLP]

** (Centang) Jenis Deep Learning: **
- [ ] Multilayer Perceptron (MLP) - untuk tabular
- [ ] Convolutional Neural Network (CNN) - untuk image
- [ ] Recurrent Neural Network (LSTM/GRU) - untuk sequential/text
- [ ] Transfer Learning - untuk image
- [ ] Transformer-based - untuk NLP
- [ ] Autoencoder - untuk unsupervised
- [ ] Neural Collaborative Filtering - untuk recommender

**Alasan Pemilihan:**  
[Mengapa arsitektur ini cocok untuk dataset Anda?]

#### 6.3.2 Arsitektur Model, Hypothesis Space & Inductive Bias  **[M]**

**Inductive bias arsitektur:** [misal: CNN → lokalitas & translation invariance; LSTM → ketergantungan sekuensial; MLP → tanpa asumsi struktur]  
**Hypothesis space:** [kapasitas model: jumlah layer, unit, parameter]

**Deskripsi Layer:**

[Jelaskan arsitektur secara detail atau buat tabel]

**Contoh:**
```
1. Input Layer: shape (224, 224, 3)
2. Conv2D: 32 filters, kernel (3,3), activation='relu'
3. MaxPooling2D: pool size (2,2)
4. Conv2D: 64 filters, kernel (3,3), activation='relu'
5. MaxPooling2D: pool size (2,2)
6. Flatten
7. Dense: 128 units, activation='relu'
8. Dropout: 0.5
9. Dense: 10 units, activation='softmax'

Total parameters: [jumlah]
Trainable parameters: [jumlah]
```

#### 6.3.3 Input & Preprocessing Khusus  **[M]**

**Input shape:** [Sebutkan dimensi input]  
**Preprocessing khusus untuk DL:**
- [Sebutkan preprocessing khusus seperti normalisasi, augmentasi, dll.]

#### 6.3.4 Optimization: Konfigurasi Training  **[O]**

**Training Configuration:**
```
- Optimizer: Adam / SGD / RMSprop
- Learning rate: [nilai]
- Loss function: [categorical_crossentropy / mse / binary_crossentropy / etc.]
- Metrics: [accuracy / mae / etc.]
- Batch size: [nilai]
- Epochs: [nilai]
- Validation split: [nilai] atau menggunakan validation set terpisah
- Callbacks: [EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, etc.]
- Regularisasi: [Dropout / L2 / BatchNorm / augmentasi]
```

**Eksperimen optimasi (minimal 2 konfigurasi dibandingkan):**

| Eksperimen | Learning Rate | Batch Size | Regularisasi | Val Loss | Val Metric |
|------------|---------------|------------|--------------|----------|------------|
| Exp-1 | [ ] | [ ] | [ ] | [ ] | [ ] |
| Exp-2 | [ ] | [ ] | [ ] | [ ] | [ ] |

#### 6.3.5 Implementasi (Ringkas)  **[M + O]**

**Framework:** TensorFlow/Keras / PyTorch
```python
# Contoh kode TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras

model_dl = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

model_dl.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model_dl.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)
```

#### 6.3.6 Training Process & Diagnosis Optimasi  **[O]**

**Training Time:**  
[Sebutkan waktu training total, misal: 15 menit]

**Computational Resource:**  
[CPU / GPU, platform: Local / Google Colab / Kaggle]

**Training History Visualization:**

[Insert plot loss dan accuracy/metric per epoch]

**Contoh visualisasi yang WAJIB:**
1. **Training & Validation Loss** per epoch
2. **Training & Validation Accuracy/Metric** per epoch

**Analisis Training:**
- Apakah model mengalami overfitting? [Ya/Tidak, jelaskan]
- Apakah model sudah converge? [Ya/Tidak, jelaskan]
- Apakah perlu lebih banyak epoch? [Ya/Tidak, jelaskan]
- Apakah learning rate sesuai (loss stabil/osilasi/lambat)? [Jelaskan]

#### 6.3.7 Model Summary
```
[Paste model.summary() output atau rangkuman arsitektur]
```

---

## 7. EVALUATION  **[P]**

### 7.1 Metrik Evaluasi

**Pilih metrik yang sesuai dengan jenis tugas:**

#### **Untuk Klasifikasi:**
- **Accuracy**: Proporsi prediksi yang benar
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean dari precision dan recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Visualisasi prediksi

#### **Untuk Regresi:**
- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **RMSE (Root Mean Squared Error)**: Akar dari MSE
- **MAE (Mean Absolute Error)**: Rata-rata absolute error
- **R² Score**: Koefisien determinasi
- **MAPE (Mean Absolute Percentage Error)**: Error dalam persentase

#### **Untuk NLP (Text Classification):**
- **Accuracy**
- **F1-Score** (terutama untuk imbalanced data)
- **Precision & Recall**
- **Perplexity** (untuk language models)

#### **Untuk Computer Vision:**
- **Accuracy**
- **IoU (Intersection over Union)** - untuk object detection/segmentation
- **Dice Coefficient** - untuk segmentation
- **mAP (mean Average Precision)** - untuk object detection

#### **Untuk Clustering:**
- **Silhouette Score**
- **Davies-Bouldin Index**
- **Calinski-Harabasz Index**

#### **Untuk Recommender System:**
- **RMSE**
- **Precision@K**
- **Recall@K**
- **NDCG (Normalized Discounted Cumulative Gain)**

**Justifikasi pemilihan metrik:** [Mengapa metrik ini selaras dengan Objective di Bab 3.2? Mengapa accuracy saja cukup/tidak cukup?]

### 7.2 Hasil Evaluasi Model (pada Test Set)

> Semua prediksi dievaluasi pada **test set** yang tidak pernah dipakai saat training maupun tuning.

#### 7.2.1 Model 1 (Baseline)

**Metrik:**
```
[Tuliskan hasil metrik, contoh:]
- Accuracy: 0.75
- Precision: 0.73
- Recall: 0.76
- F1-Score: 0.74
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

#### 7.2.2 Model 2 (Advanced/ML)

**Metrik:**
```
- Accuracy: 0.85
- Precision: 0.84
- Recall: 0.86
- F1-Score: 0.85
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Feature Importance (jika applicable):**  
[Insert plot feature importance untuk tree-based models]

#### 7.2.3 Model 3 (Deep Learning)

**Metrik:**
```
- Accuracy: 0.89
- Precision: 0.88
- Recall: 0.90
- F1-Score: 0.89
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Training History:**  
[Sudah diinsert di Section 6.3.6]

**Test Set Predictions:**  
[Tampilkan minimal 5 contoh prediksi benar dan salah]

### 7.3 Perbandingan Ketiga Model

**Tabel Perbandingan:**

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time |
|-------|----------|-----------|--------|----------|---------------|----------------|
| Baseline (Model 1) | 0.75 | 0.73 | 0.76 | 0.74 | 2s | 0.01s |
| Advanced (Model 2) | 0.85 | 0.84 | 0.86 | 0.85 | 30s | 0.05s |
| Deep Learning (Model 3) | 0.89 | 0.88 | 0.90 | 0.89 | 15min | 0.1s |

**Visualisasi Perbandingan:**  
[Insert bar chart atau plot perbandingan metrik]

### 7.4 Analisis Hasil

**Interpretasi:**

1. **Model Terbaik:**  
   [Sebutkan model mana yang terbaik dan mengapa]

2. **Perbandingan dengan Baseline:**  
   [Jelaskan peningkatan performa dari baseline ke model lainnya]

3. **Trade-off:**  
   [Jelaskan trade-off antara performa vs kompleksitas vs waktu training]

4. **Error Analysis:**  
   [Jelaskan jenis kesalahan yang sering terjadi, kasus yang sulit diprediksi]

5. **Overfitting/Underfitting & Generalization Gap:**  
   [Bandingkan performa train vs validation vs test; analisis apakah model overfitting atau underfitting]

6. **Kesesuaian Inductive Bias:**  
   [Apakah asumsi tiap model (Bab 6.x.2) terbukti sesuai dengan data? Jelaskan dengan bukti hasil]

### 7.5 Umpan Balik Lifecycle (Iterasi L → M → O)

Jelaskan minimal **satu** iterasi perbaikan berdasarkan hasil evaluasi:

| Temuan dari Evaluasi [P] | Tahap yang Diperbaiki | Tindakan | Dampak pada Metrik |
|--------------------------|-----------------------|----------|--------------------|
| [misal: recall kelas minoritas rendah] | [L] data | [misal: class weight / SMOTE] | [ ... ] |
| [misal: gap train-val besar] | [O] optimasi | [misal: tambah dropout, early stopping] | [ ... ] |
| [misal: model terlalu sederhana] | [M] desain | [misal: tambah layer / ganti arsitektur] | [ ... ] |

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama

**Model Terbaik:**  
[Sebutkan model terbaik berdasarkan evaluasi]

**Alasan:**  
[Jelaskan mengapa model tersebut lebih unggul]

**Pencapaian Goals:**  
[Apakah goals di Section 3.3 tercapai? Jelaskan]

### 8.2 Key Insights

**Insight dari Data:**
- [Insight 1]
- [Insight 2]
- [Insight 3]

**Insight dari Modeling:**
- [Insight 1]
- [Insight 2]

### 8.3 Ringkasan per Tahap Lifecycle

| Tahap | Keputusan Kunci | Pelajaran |
|-------|-----------------|-----------|
| **L** – Learning Problem | [ ... ] | [ ... ] |
| **M** – Model Design | [ ... ] | [ ... ] |
| **O** – Optimization | [ ... ] | [ ... ] |
| **P** – Predict & Evaluate | [ ... ] | [ ... ] |

### 8.4 Kontribusi Proyek

**Manfaat praktis:**  
[Jelaskan bagaimana proyek ini dapat digunakan di dunia nyata]

**Kontribusi terhadap SDGs:**  
[Jelaskan secara jujur dan terukur bagaimana hasil proyek mendukung target SDG yang dipilih (Bab 2.2), serta keterbatasan kontribusinya]

**Pembelajaran yang didapat:**  
[Jelaskan apa yang Anda pelajari dari proyek ini]

---

## 9. FUTURE WORK (Opsional)

Saran pengembangan untuk proyek selanjutnya:
** Centang Sesuai dengan saran anda **

**Data:**
- [ ] Mengumpulkan lebih banyak data
- [ ] Menambah variasi data
- [ ] Feature engineering lebih lanjut

**Model:**
- [ ] Mencoba arsitektur DL yang lebih kompleks
- [ ] Hyperparameter tuning lebih ekstensif
- [ ] Ensemble methods (combining models)
- [ ] Transfer learning dengan model yang lebih besar

**Deployment:**
- [ ] Membuat API (Flask/FastAPI)
- [ ] Membuat web application (Streamlit/Gradio)
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud (Heroku, GCP, AWS)

**Optimization:**
- [ ] Model compression (pruning, quantization)
- [ ] Improving inference speed
- [ ] Reducing model size

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository

**Link Repository:** [URL GitHub Anda]

**Repository harus berisi:**
- ✅ Notebook Jupyter/Colab dengan hasil running
- ✅ Script Python (jika ada)
- ✅ requirements.txt atau environment.yml
- ✅ README.md yang informatif
- ✅ Folder structure yang terorganisir
- ✅ .gitignore (jangan upload dataset besar)

**Saran struktur folder (mengikuti lifecycle):**
```
project/
├── data/                 # [L] dataset (jangan upload jika besar)
├── notebooks/
│   ├── 01_learning_problem_eda.ipynb      # [L]
│   ├── 02_data_preparation.ipynb          # [L]
│   ├── 03_model_design_baseline_ml.ipynb  # [M]
│   ├── 04_deep_learning_optimization.ipynb# [M+O]
│   └── 05_evaluation_comparison.ipynb     # [P]
├── src/                  # script modular
├── models/               # model tersimpan
├── reports/figures/
├── requirements.txt
└── README.md
```

### 10.2 Environment & Dependencies

**Python Version:** [3.8 / 3.9 / 3.10 / 3.11]

**Main Libraries & Versions:**
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# Deep Learning Framework (pilih salah satu)
tensorflow==2.14.0  # atau
torch==2.1.0        # PyTorch

# Additional libraries (sesuaikan)
xgboost==1.7.6
lightgbm==4.0.0
opencv-python==4.8.0  # untuk computer vision
nltk==3.8.1           # untuk NLP
transformers==4.30.0  # untuk BERT, dll
```

### 10.3 Reproducibility Checklist
- [ ] Random seed ditetapkan (misal 42) di semua library
- [ ] Notebook dapat dijalankan ulang dari awal sampai akhir
- [ ] Test set tidak dipakai saat training/tuning
- [ ] Versi library tercatat

---

## 11. TIMELINE PENGERJAAN (6 MINGGU)

Pengerjaan mengikuti urutan lifecycle L → M → O → P dengan satu iterasi perbaikan. Rincian tugas per minggu ada di file panduan terpisah: **Panduan_Tahapan_6_Minggu.md**.

| Minggu | Fokus Lifecycle | Aktivitas | Deliverable / Checkpoint |
|--------|-----------------|-----------|--------------------------|
| **1** | **L** – Learning Problem | Mahasiswa mengajukan ide sendiri selaras dengan satu SDG, pilih dataset, studi literatur, rumuskan problem statement, target, objective, data (Bab 1–3) | Formulir ide + proposal singkat 1–2 halaman; ide & dataset disetujui dosen. **Checkpoint 1** |
| **2** | **L** – Data | Data understanding, EDA (opsional), cleaning, transformasi, splitting (Bab 4–5) | Notebook data preparation + repo GitHub dibuat |
| **3** | **M** – Model Design | Rancang & latih Model 1 (baseline) dan Model 2 (advanced); jelaskan hypothesis space & inductive bias (Bab 6.1–6.2) | Hasil baseline & advanced pada validation set. **Checkpoint 2** |
| **4** | **M → O** | Tuning Model 2; desain arsitektur Model 3 (deep learning) dan training pertama (Bab 6.2.4, 6.3.1–6.3.5) | Model DL berjalan end-to-end (training awal) |
| **5** | **O** – Optimization | Optimasi Model 3: eksperimen learning rate/batch size/regularisasi, plot loss/metric, diagnosis konvergensi (Bab 6.3.4–6.3.7) | Model DL terlatih (≥10 epoch) + tabel eksperimen. **Checkpoint 3** |
| **6** | **P** – Predict & Evaluate | Evaluasi test set, perbandingan 3 model, error analysis, iterasi perbaikan, kesimpulan, video, finalisasi repo (Bab 7–10) | Laporan final + GitHub + video + presentasi |

**Catatan untuk dosen:**
- Enam minggu memberi ruang tambahan untuk tahap Optimization, tempat mahasiswa biasanya paling banyak mengalami kendala (training deep learning, tuning, overfitting).
- Tiga checkpoint (akhir minggu 1, 3, dan 5) membantu mendeteksi keterlambatan lebih awal.

---

## 12. RUBRIK PENILAIAN (USULAN)

| Komponen | Tahap | Bobot |
|----------|-------|-------|
| Orisinalitas ide mahasiswa & keterkaitan dengan SDGs (Bab 2.2) | L | 10% |
| Problem formulation (target, objective, data), latar belakang & referensi | L | 10% |
| Data understanding & data preparation (tanpa data leakage) | L | 10% |
| Model design: baseline & advanced + justifikasi hypothesis space/inductive bias | M | 15% |
| Model deep learning: arsitektur, training ≥10 epoch, plot loss/metric | M + O | 15% |
| Optimization: tuning hyperparameter, regularisasi, perbandingan eksperimen | O | 10% |
| Evaluasi: metrik relevan, perbandingan 3 model, error analysis, iterasi lifecycle | P | 20% |
| Reproducibility: GitHub, README, requirements, struktur folder | – | 5% |
| Video pembahasan & kualitas penulisan laporan | – | 5% |
| **Total** | | **100%** |

> Bobot dapat disesuaikan dengan kebijakan mata kuliah.

---

## LAMPIRAN A. SDGs SEBAGAI SUMBER IDE

Referensi resmi: [Transforming our world: the 2030 Agenda for Sustainable Development](https://sdgs.un.org/2030agenda)

**Aturan ide:**
- Ide **harus berasal dari mahasiswa**, bukan dari daftar contoh di bawah. Contoh hanya inspirasi.
- Pilih **satu** SDG utama dan sebutkan target (nomor target) yang relevan.
- Masalah harus nyata, spesifik, dan dapat dirumuskan sebagai tugas ML (klasifikasi, regresi, forecasting, dll.).
- Dataset harus tersedia (publik atau dikumpulkan sendiri secara etis) dan cukup untuk melatih tiga model, termasuk deep learning.
- Disarankan setiap mahasiswa/kelompok memiliki ide yang berbeda; dosen memeriksa di Checkpoint 1.

| SDG | Nama (Indonesia) | Contoh arah ide ML (inspirasi) |
|-----|------------------|--------------------------------|
| 1 | Tanpa Kemiskinan | Prediksi tingkat kemiskinan wilayah dari data sosial-ekonomi (tabular) |
| 2 | Tanpa Kelaparan | Klasifikasi penyakit daun tanaman (image); prediksi hasil panen (tabular) |
| 3 | Kehidupan Sehat dan Sejahtera | Deteksi penyakit dari data klinis atau citra medis; prediksi risiko diabetes |
| 4 | Pendidikan Berkualitas | Prediksi risiko putus studi/performa siswa; analisis ulasan pembelajaran (teks) |
| 5 | Kesetaraan Gender | Deteksi ujaran kebencian atau bias gender pada teks |
| 6 | Air Bersih dan Sanitasi Layak | Prediksi kelayakan/kualitas air minum (tabular) |
| 7 | Energi Bersih dan Terjangkau | Forecasting konsumsi atau produksi energi (time series) |
| 8 | Pekerjaan Layak dan Pertumbuhan Ekonomi | Klasifikasi lowongan kerja atau prediksi pengangguran |
| 9 | Industri, Inovasi, dan Infrastruktur | Deteksi kerusakan jalan (image); predictive maintenance |
| 10 | Berkurangnya Kesenjangan | Pengenalan bahasa isyarat (image); analisis akses layanan publik |
| 11 | Kota dan Permukiman yang Berkelanjutan | Prediksi kualitas udara kota; prediksi kepadatan lalu lintas |
| 12 | Konsumsi dan Produksi yang Bertanggung Jawab | Klasifikasi jenis sampah (image); prediksi limbah makanan |
| 13 | Penanganan Perubahan Iklim | Prediksi curah hujan/suhu; deteksi titik api atau kebakaran hutan |
| 14 | Ekosistem Lautan | Klasifikasi spesies ikan; deteksi sampah laut atau kesehatan terumbu karang |
| 15 | Ekosistem Daratan | Klasifikasi spesies flora/fauna; deteksi deforestasi dari citra |
| 16 | Perdamaian, Keadilan, dan Kelembagaan yang Tangguh | Deteksi hoaks/berita palsu (teks); deteksi fraud |
| 17 | Kemitraan untuk Mencapai Tujuan | Sistem rekomendasi kolaborasi/mitra; analisis data terbuka pemerintah |

---

## LAMPIRAN B. FORMULIR PENGAJUAN IDE (DIKUMPULKAN PADA CHECKPOINT 1)

| Komponen | Isian Mahasiswa |
|----------|-----------------|
| Nama / NIM | [ ... ] |
| Judul ide | [ ... ] |
| SDG & target yang dipilih | [ ... ] |
| Masalah nyata yang diangkat | [ ... ] |
| Tugas ML (Target & jenis tugas) | [ ... ] |
| Objective & metrik utama | [ ... ] |
| Dataset (sumber, URL, ukuran, jenis data) | [ ... ] |
| Rencana tiga model (baseline / advanced / deep learning) | [ ... ] |
| Risiko & pertimbangan etika (bias, privasi) | [ ... ] |
| Referensi awal (minimal 2) | [ ... ] |

**Kriteria persetujuan dosen:**
- [ ] Ide berasal dari mahasiswa dan tidak duplikat
- [ ] Selaras jelas dengan satu SDG dan targetnya
- [ ] Dapat dirumuskan sebagai masalah ML dengan target yang jelas
- [ ] Dataset tersedia, legal/etis digunakan, dan cukup besar
- [ ] Layak untuk tiga model termasuk deep learning dalam 6 minggu
