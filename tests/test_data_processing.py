import sys
import os
import pandas as pd
import numpy as np
import joblib

# permettre d'importer depuis src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_processing import optimize_memory


# ---------------------------------------------------
# Test 1 : vérifier la gestion des valeurs manquantes
# ---------------------------------------------------
def test_missing_values_handling():

    df = pd.DataFrame({
        "age": [25, 30, np.nan, 40],
        "weight": [70, np.nan, 80, 90]
    })

    df_filled = df.fillna(df.mean())

    assert df_filled.isnull().sum().sum() == 0


# ---------------------------------------------------
# Test 2 : vérifier la fonction optimize_memory
# ---------------------------------------------------
def test_optimize_memory():

    df = pd.DataFrame({
        "A": np.random.randint(0, 100, 1000),
        "B": np.random.rand(1000)
    })

    df_optimized = optimize_memory(df)

    assert isinstance(df_optimized, pd.DataFrame)


# ---------------------------------------------------
# Test 3 : vérifier chargement modèle + prédiction
# ---------------------------------------------------
def test_model_prediction():

    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
    )

    if os.path.exists(model_path):

        model = joblib.load(model_path)

        X_test = pd.DataFrame([[25, 70, 3, 2, 1]])

        prediction = model.predict(X_test)

        assert prediction is not None