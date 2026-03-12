import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ==========================
# Load dataset
# ==========================

data_path = r"C:\Users\Zouhire\Documents\GitHub\PROJET___-2__-GROUPE__7__CODING__WEEK-1\data_clean.csv"
df = pd.read_csv(data_path)

print("Columns:", df.columns)

# Target = dernière colonne
target = df.columns[-1]

X = df.drop(target, axis=1)
y = df[target]

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Models
# ==========================

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="mlogloss"),
    "LightGBM": LGBMClassifier()
}

best_model = None
best_score = 0
best_name = ""

# ==========================
# Training
# ==========================

for name, model in models.items():

    print("\nTraining", name)

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)

    roc_auc = roc_auc_score(
        y_test,
        y_proba,
        multi_class="ovr"
    )

    print("ROC-AUC:", roc_auc)

    if roc_auc > best_score:
        best_score = roc_auc
        best_model = model
        best_name = name

# ==========================
# Save model
# ==========================

joblib.dump(best_model, "best_model.pkl")

print("\nBest model:", best_name)
print("Best ROC-AUC:", best_score)
print("Model saved: best_model.pkl")