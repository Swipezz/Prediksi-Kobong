from pathlib import Path
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix

# Otomatis mendeteksi folder root proyek (Prediksi-Kobong)
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data hasil preprocessing Minggu 2 (HANYA Train & Validation)
X_train = pd.read_csv(DATA_DIR / "X_train_ready.csv")
y_train = pd.read_csv(DATA_DIR / "y_train_ready.csv").values.ravel()
X_val = pd.read_csv(DATA_DIR / "X_val_ready.csv")
y_val = pd.read_csv(DATA_DIR / "y_val_ready.csv").values.ravel()

print(f"Train shape: {X_train.shape} | Validation shape: {X_val.shape}")

# 2. Inisialisasi & Latih Baseline Model (Parameter Awal)
dt_baseline = DecisionTreeClassifier(random_state=42)
dt_baseline.fit(X_train, y_train)

# 3. Prediksi & Evaluasi pada Validation Set
y_val_pred = dt_baseline.predict(X_val)
y_val_proba = dt_baseline.predict_proba(X_val)[:, 1]

print("\n=== HASIL VALIDASI: BASELINE (DECISION TREE) ===")
print(f"Accuracy : {accuracy_score(y_val, y_val_pred):.4f}")
print(f"F1-Score : {f1_score(y_val, y_val_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_val, y_val_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

# 4. Simpan model baseline
joblib.dump(dt_baseline, MODELS_DIR / "baseline_decision_tree.pkl")