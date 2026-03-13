import pandas as pd
import sys
import os
import joblib
import numpy as np

# ajouter le dossier src au path pour importer les fonctions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_processing import optimize_memory


# ---------------------------------------------------
# TEST 1 : verify missing values handling
# ---------------------------------------------------

def test_missing_values_handling():

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_clean.csv')

    df = pd.read_csv(data_path)

    # vérifier qu'il n'y a pas de valeurs manquantes
    assert df.isnull().sum().sum() == 0


# ---------------------------------------------------
# TEST 2 : verify optimize_memory(df) function
# ---------------------------------------------------

def test_optimize_memory():

    df = pd.DataFrame({
        "Age": [20, 30, 40],
        "Height": [1.70, 1.80, 1.65],
        "Weight": [70, 80, 60]
    })

    before = df.memory_usage().sum()

    df_optimized = optimize_memory(df)

    after = df_optimized.memory_usage().sum()

    # vérifier que la mémoire a diminué ou reste égale
    assert after <= before


# ---------------------------------------------------
# TEST 3 : verify model loading and prediction
# ---------------------------------------------------

def test_model_loading_and_prediction():

    model_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.pkl')
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_clean.csv')

    # charger le modèle
    model = joblib.load(model_path)

    # charger le dataset
    df = pd.read_csv(data_path)

    # séparer features et target
    X = df.drop("NObeyesdad", axis=1)

    # prendre une ligne du dataset
    sample = X.iloc[:1]

    # faire une prédiction
    prediction = model.predict(sample)

    # vérifier que la prédiction existe
    assert prediction is not None