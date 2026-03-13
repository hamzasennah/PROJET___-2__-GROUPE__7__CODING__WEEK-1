import pandas as pd
import numpy as np
import sys
import os
import joblib
import subprocess

# ajouter src au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_processing import optimize_memory


# --------------------------------------------------
# TEST 1 : verify missing values handling
# --------------------------------------------------

def test_missing_values_handling():

    print("\n===== TEST : Missing Values Handling =====")

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_clean.csv')

    df = pd.read_csv(data_path)

    missing_values = df.isnull().sum().sum()

    print("Total missing values:", missing_values)

    assert missing_values == 0


# --------------------------------------------------
# TEST 2 : verify optimize_memory(df)
# --------------------------------------------------

def test_optimize_memory():

    print("\n===== TEST : optimize_memory(df) =====")

    df = pd.DataFrame({
        "Age": [21, 35, 40],
        "Height": [1.70, 1.80, 1.65],
        "Weight": [70, 85, 60]
    })

    before = df.memory_usage().sum()
    print("Memory BEFORE optimization:", before)

    df_opt = optimize_memory(df)

    after = df_opt.memory_usage().sum()
    print("Memory AFTER optimization:", after)

    assert after <= before


# --------------------------------------------------
# TEST 3 : verify model loading + characteristics
# --------------------------------------------------

def test_model_loading_and_prediction():

    print("\n===== TEST : Model Loading =====")

    model_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.pkl')

    model = joblib.load(model_path)

    print("Model loaded successfully :", type(model))

    # sample data pour prédiction
    sample = np.array([[21,1.70,70,1,1,2,3,1,0,2,0,1,1,0,1]])

    prediction = model.predict(sample)

    print("Prediction result:", prediction)

    assert prediction is not None

    # --------------------------------------------------
    # afficher les performances du modèle
    # --------------------------------------------------

    print("\n===== MODEL PERFORMANCE =====")

    eval_script = os.path.join(os.path.dirname(__file__), '..', 'src', 'evaluate_model.py')

    result = subprocess.run(
        ["python", eval_script],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    assert result.returncode == 0