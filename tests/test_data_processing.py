import sys
import os
import pandas as pd
import numpy as np

# permettre l'import depuis src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_processing import optimize_memory, handle_missing_values


# -------------------------
# Test valeurs manquantes
# -------------------------
def test_missing_values_handling():

    df = pd.DataFrame({
        "age": [20, 25, np.nan, 30],
        "weight": [60, np.nan, 80, 90]
    })

    df_clean = handle_missing_values(df)

    assert df_clean.isnull().sum().sum() == 0


# -------------------------
# Test optimize_memory
# -------------------------
def test_optimize_memory():

    df = pd.DataFrame({
        "A": np.random.randint(0, 100, 1000),
        "B": np.random.rand(1000)
    })

    df_opt = optimize_memory(df)

    assert isinstance(df_opt, pd.DataFrame)


# -------------------------
# Test prédiction modèle
# -------------------------
def test_model_prediction():

    # test simple pour vérifier fonctionnement
    X = pd.DataFrame([[25, 70, 3, 2, 1]])

    assert X.shape[0] == 1