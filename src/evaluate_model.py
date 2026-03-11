import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# =========================
# Charger données test
# =========================

X_test, y_test = joblib.load("models/test_data.pkl")

# =========================
# Charger modèles
# =========================

rf = joblib.load("models/random_forest.pkl")
xgb_model = joblib.load("models/xgboost.pkl")
lgb_model = joblib.load("models/lightgbm.pkl")

models = {
    "Random Forest": rf,
    "XGBoost": xgb_model,
    "LightGBM": lgb_model
}

results = []

# =========================
# Evaluation
# =========================

for name, model in models.items():

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc
    })

df = pd.DataFrame(results)

print("\nComparaison des modèles :")
print(df)

df.to_csv("models/model_results.csv", index=False)

best_model = df.sort_values("ROC-AUC", ascending=False).iloc[0]

print("\nMeilleur modèle :", best_model["Model"])