import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# =========================
# Charger dataset Excel
# =========================

data = pd.read_excel("C:/Users/Zouhire/zouhjbn/mon_env/ObesityDataSet_cleaned.xlsx")

print("Dataset chargé avec succès")
print(data.head())

# =========================
# Encoder les colonnes texte
# =========================

for column in data.select_dtypes(include=['object']).columns:
    data[column] = LabelEncoder().fit_transform(data[column])

# =========================
# Target column
# =========================

target = "NObeyesdad"

X = data.drop(target, axis=1)
y = data[target]

# =========================
# Train / Test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Créer dossier models
# =========================

os.makedirs("models", exist_ok=True)

# =========================
# Random Forest
# =========================

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# =========================
# XGBoost
# =========================

xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# =========================
# LightGBM
# =========================

lgb_model = lgb.LGBMClassifier(n_estimators=200)
lgb_model.fit(X_train, y_train)

# =========================
# Sauvegarder modèles
# =========================

joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(xgb_model, "models/xgboost.pkl")
joblib.dump(lgb_model, "models/lightgbm.pkl")

joblib.dump((X_test, y_test), "models/test_data.pkl")

print("Training terminé. Modèles sauvegardés.")