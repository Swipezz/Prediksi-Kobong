# 🔥 Prediksi-Kobong: Sistem Prediksi Risiko Titik Api Harian (H+1) untuk Peringatan Dini Karhutla

Proyek *Machine Learning* dan *Deep Learning* berbasis kerangka **ML Lifecycle (L–M–O–P)** untuk memprediksi potensi kemunculan titik api (*hotspot*) harian pada hari berikutnya (`H+1`) di wilayah rawan kebakaran hutan dan lahan (karhutla) Sumatra dan Kalimantan.

* **Tim Pengembang:** Adam Noverian (244311033) & Fachrozzi Rizky Wibowo (244311041)
* **Program Studi:** Teknologi Rekayasa Perangkat Lunak — Politeknik Negeri Madiun
* **Mata Kuliah:** Data Science (Dosen Pengampu: Gus Nanang Syaifuddiin, S.Kom., M.Kom.)
* **Keterkaitan SDGs:** SDG 13 (*Climate Action*) & SDG 15 (*Life on Land*)

---

## 📌 Rumusan Masalah & Target Prediksi

* **Target (`fire_next_day`):** Klasifikasi biner (`1` = muncul $\ge 1$ *hotspot* VIIRS jenis kebakaran vegetasi dengan tingkat kepercayaan *nominal/high* dalam radius 25 km pada hari $t+1$; `0` = aman).
* **Cakupan Spasiotemporal:** 6 lokasi studi representatif di lahan gambut/kering Sumatra Selatan (`L1`–`L3`) dan Kalimantan (`L4`–`L6`) pada rentang tahun **2019–2024** (`12.912` sampel valid).
* **Metrik Evaluasi Utama:** **PR-AUC** dan **F1-Score** (didampingi *ROC-AUC* dan *Recall*), karena proporsi kelas positif minoritas dan bervariasi antar-tahun (10,0% pada data latih vs 20,7% pada data validasi).

---

## 📂 Struktur Direktori Proyek

```text
Prediksi-Kobong/
├── data/
│   ├── raw/                                 # Dataset mentah cuaca Open-Meteo & satelit VIIRS (diabaikan Git)
│   └── processed/                           # Dataset bersih hasil splitting & transformasi
│       ├── train_model.csv                  # 8.586 sampel (2019–2022 | 10,0% positif)
│       ├── val_model.csv                    # 2.148 sampel (2023 | 20,7% positif)
│       ├── test_model.csv                   # 2.178 sampel (2024 | dikunci hingga Minggu 6)
│       ├── feature_lists.json               # Daftar 28 fitur model & grup fitur ablation
│       ├── class_weights.json               # Bobot kelas terhitung dari train (scale_pos_weight: 9.007)
│       └── test_checksums.json              # Verifikasi integritas SHA-256 test set
│
├── notebooks/
│   ├── 01_learning_problem_eda.ipynb        # Minggu 1–2: Perumusan masalah & EDA (6 visualisasi)
│   ├── 02_data_preparation.ipynb            # Minggu 2: Cleaning, feature engineering, split kronologis, scaling
│   ├── 03_model_design_baseline_ml.ipynb    # Minggu 3: Baseline Decision Tree + uji ablation
│   └── 03_model_design_advanced_ml.ipynb    # Minggu 3: SVM (RBF 5-Fold) & Deep Learning MLP (128-64-32)
│
├── src/
│   ├── ambil_cuaca.py                       # Pengunduhan data meteorologi historis Open-Meteo
│   └── prep.py                              # Modul pipeline pemrosesan fitur, label H+1, & splitting
│
├── models/                                  # Artefak preprocessor & model terlatih
│   ├── preprocessor.joblib
│   ├── baseline_decision_tree.pkl
│   ├── advanced_svm_rbf_5fold.pkl
│   └── advanced_mlp_full.pt
│
├── reports/
│   ├── data_prep_summary.json
│   └── figures/                             # Grafik visualisasi EDA & distribusi split (fig01..fig07)
│
├── EAS_Laporan_Proyek_ML_LMOP.md            # Dokumen laporan utama L-M-O-P
├── .gitignore
├── requirements.txt
└── README.md
