# save_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Exemple : adapte selon ton modèle
from sklearn.model_selection import train_test_split
import pickle

# --- Charger les données ---
data_path = r"C:\Users\Zouhire\Documents\GitHub\PROJET___-2__-GROUPE__7__CODING__WEEK-1\data_clean.csv"
df = pd.read_csv(data_path)

# --- Préparer X et y ---
# Remplace 'target' par le nom de la colonne à prédire
X = df.drop(columns=['target'])
y = df['target']

# Encodage simple pour les colonnes catégorielles
X_encoded = pd.get_dummies(X)

# --- Split pour entraînement (optionnel) ---
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# --- Créer le modèle ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Sauvegarder le modèle en pickle binaire ---
model_path = r"C:\Users\Zouhire\Documents\GitHub\PROJET___-2__-GROUPE__7__CODING__WEEK-1\best_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("Modèle sauvegardé avec succès dans best_model.pkl !")