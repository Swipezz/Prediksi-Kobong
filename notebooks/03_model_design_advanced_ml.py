import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.svm import SVC

# 1. Konfigurasi Seed & Path Otomatis
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
)
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load Data Train & Validation (100% Utuh)
X_train = pd.read_csv(DATA_DIR / "X_train_ready.csv").values
y_train = pd.read_csv(DATA_DIR / "y_train_ready.csv").values.ravel()
X_val_df = pd.read_csv(DATA_DIR / "X_val_ready.csv")  # DataFrame untuk Decision Tree
X_val = X_val_df.values                                # NumPy array untuk SVM & MLP
y_val = pd.read_csv(DATA_DIR / "y_val_ready.csv").values.ravel()

print(f"Data dimuat (100% Utuh) -> Train: {X_train.shape} | Validation: {X_val.shape}\n")

# ==========================================
# MODEL BARU: SVM (RBF) 5-FOLD CALIBRATED
# ==========================================
print("Memulai pelatihan SVM (RBF) 5-Fold pada 100% data...")
print("(Log titik-titik [LibSVM] akan muncul dari proses paralel di latar belakang)\n")
start_svm = time.time()

# verbose=True diletakkan di SVC (bukan di CalibratedClassifierCV)
# cache_size=1000 per worker agar total cache 5 proses paralel aman di RAM (~5 GB)
base_svc = SVC(kernel="rbf", cache_size=1000, verbose=True, random_state=42)
svm_5fold = CalibratedClassifierCV(base_svc, cv=5, ensemble=False, n_jobs=-1)
svm_5fold.fit(X_train, y_train)

durasi_5fold = time.time() - start_svm
print(f"\n[SELESAI] SVM 5-Fold rampung dalam {durasi_5fold / 60:.2f} menit ({durasi_5fold:.1f} detik).\n")

# Simpan ke nama file baru agar tidak menimpa advanced_svm_rbf_full.pkl
joblib.dump(svm_5fold, MODELS_DIR / "advanced_svm_rbf_5fold.pkl")

y_val_pred_svm5 = svm_5fold.predict(X_val)
y_val_proba_svm5 = svm_5fold.predict_proba(X_val)[:, 1]

# ==========================================
# MUAT 3 MODEL SEBELUMNYA DARI FOLDER MODELS/
# ==========================================
# 1. Decision Tree (Baseline)
dt_baseline = joblib.load(MODELS_DIR / "baseline_decision_tree.pkl")
y_val_pred_dt = dt_baseline.predict(X_val_df)
y_val_proba_dt = dt_baseline.predict_proba(X_val_df)[:, 1]

# 2. SVM Exact 1x Fit (Eksperimen 39.8 menit sebelumnya)
svm_exact = joblib.load(MODELS_DIR / "advanced_svm_rbf_full.pkl")
y_val_pred_svm1 = svm_exact.predict(X_val)
y_val_score_svm1 = svm_exact.decision_function(X_val)

# 3. MLP 128-64-32 (Eksperimen 3.3 menit sebelumnya)
class TabularMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


mlp_model = TabularMLP(X_train.shape[1])
mlp_model.load_state_dict(torch.load(MODELS_DIR / "advanced_mlp_full.pt", map_location="cpu", weights_only=True))
mlp_model.eval()
with torch.no_grad():
    logits = mlp_model(torch.tensor(X_val, dtype=torch.float32))
    y_val_proba_mlp = torch.sigmoid(logits).numpy().ravel()
    y_val_pred_mlp = (y_val_proba_mlp >= 0.5).astype(int)

# ==========================================
# TABEL PERBANDINGAN 4 MODEL PADA VALIDATION SET
# ==========================================
comparison_df = pd.DataFrame(
    {
        "Model": [
            "1. Baseline: Decision Tree",
            "2. SVM (RBF - 1x Exact Fit)",
            "3. SVM (RBF - 5-Fold Calibrated)",
            "4. Deep Learning: MLP (128-64-32)",
        ],
        "Accuracy": [
            accuracy_score(y_val, y_val_pred_dt),
            accuracy_score(y_val, y_val_pred_svm1),
            accuracy_score(y_val, y_val_pred_svm5),
            accuracy_score(y_val, y_val_pred_mlp),
        ],
        "F1-Score": [
            f1_score(y_val, y_val_pred_dt),
            f1_score(y_val, y_val_pred_svm1),
            f1_score(y_val, y_val_pred_svm5),
            f1_score(y_val, y_val_pred_mlp),
        ],
        "ROC-AUC": [
            roc_auc_score(y_val, y_val_proba_dt),
            roc_auc_score(y_val, y_val_score_svm1),
            roc_auc_score(y_val, y_val_proba_svm5),
            roc_auc_score(y_val, y_val_proba_mlp),
        ],
    }
)

print("=== PERBANDINGAN SKOR VALIDASI (EXACT VS 5-FOLD) ===")
print(comparison_df.to_string(index=False))
print("\n--- Detail Classification Report SVM 5-Fold ---")
print(classification_report(y_val, y_val_pred_svm5))