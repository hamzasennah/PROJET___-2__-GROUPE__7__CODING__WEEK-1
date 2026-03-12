"""
Tests unitaires pour le module de traitement des données et le modèle.
Exécution : pytest tests/test_data_processing.py -v
"""

import pytest
import pandas as pd
import numpy as np
import os
import pickle
import joblib
from pandas.testing import assert_frame_equal

# Import des fonctions depuis src.data_processing
from src.data_processing import handle_missing_values, optimize_memory

# Fonction utilitaire pour charger un modèle (pickle ou joblib)
def load_model(model_path):
    """Charge un modèle pickle ou joblib."""
    if model_path.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        return joblib.load(model_path)

class TestDataProcessing:
    """Tests pour les fonctions de traitement des données."""

    def test_handle_missing_values_drop(self):
        """Vérifie que les lignes avec valeurs manquantes sont supprimées."""
        data = {
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8],
            'C': ['x', 'y', 'z', np.nan]
        }
        df = pd.DataFrame(data)
        df_clean = handle_missing_values(df, strategy='drop')
        # Après drop, seule la ligne sans NaN doit rester (index 0)
        expected = pd.DataFrame({
            'A': [1],
            'B': [5],
            'C': ['x']
        }, index=[0])
        assert_frame_equal(df_clean.reset_index(drop=True), expected)

    def test_handle_missing_values_fill(self):
        """Vérifie que les valeurs manquantes sont imputées correctement."""
        data = {
            'num': [1.0, 2.0, np.nan, 4.0],
            'cat': ['a', 'b', np.nan, 'a']
        }
        df = pd.DataFrame(data)
        df_filled = handle_missing_values(df, strategy='fill')
        # Moyenne de 'num' = (1+2+4)/3 = 7/3, mode de 'cat' = 'a'
        expected = pd.DataFrame({
            'num': [1.0, 2.0, 7/3, 4.0],
            'cat': ['a', 'b', 'a', 'a']
        })
        pd.testing.assert_frame_equal(df_filled, expected, check_dtype=False)

    def test_optimize_memory_reduces_size(self):
        """Vérifie que l'optimisation mémoire réduit l'empreinte."""
        df = pd.DataFrame({
            'small_int': [1, 2, 3, 4],          # peut tenir en int8
            'large_int': [1000, 2000, 3000, 4000], # int16
            'float_col': [1.5, 2.5, 3.5, 4.5]   # float32
        })
        mem_before = df.memory_usage(deep=True).sum()
        df_opt = optimize_memory(df)
        mem_after = df_opt.memory_usage(deep=True).sum()
        assert mem_after < mem_before, "L'optimisation mémoire n'a pas réduit la taille"
        # Vérifier les types convertis
        assert df_opt['small_int'].dtype == np.int8
        assert df_opt['large_int'].dtype == np.int16
        assert df_opt['float_col'].dtype == np.float32

    def test_optimize_memory_preserves_data(self):
        """Vérifie que l'optimisation ne change pas les valeurs."""
        df = pd.DataFrame({
            'int': [1, 2, 3],
            'float': [1.1, 2.2, 3.3]
        })
        df_opt = optimize_memory(df)
        assert_frame_equal(df, df_opt, check_dtype=False)

class TestModel:
    """Tests pour le chargement et la prédiction du modèle."""

    @pytest.fixture
    def sample_model_path(self, tmp_path):
        """Crée un modèle factice pour les tests."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        model = LogisticRegression()
        model.fit(X, y)
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model_path

    def test_model_loading(self, sample_model_path):
        """Vérifie que le modèle se charge sans erreur."""
        model = load_model(sample_model_path)
        assert model is not None
        assert hasattr(model, 'predict')

    def test_model_prediction(self, sample_model_path):
        """Vérifie que la prédiction retourne un résultat cohérent."""
        model = load_model(sample_model_path)
        # Créer un échantillon de 4 features (comme le modèle factice)
        sample = pd.DataFrame([[0.5, -1.2, 3.4, 2.1]])
        pred = model.predict(sample)
        assert isinstance(pred, np.ndarray)
        assert len(pred) == 1
        if hasattr(model, 'classes_'):
            assert pred[0] in model.classes_

    def test_model_prediction_with_real_file(self):
        """Test avec le vrai modèle (si disponible). À adapter avec vos chemins."""
        model_path = r"C:\Users\Zouhire\Documents\GitHub\PROJET___-2__-GROUPE__7__CODING__WEEK-1\best_model.pkl"
        if not os.path.exists(model_path):
            pytest.skip("Fichier modèle non trouvé, test ignoré")
        model = load_model(model_path)
        # Récupérer les noms des features si possible
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        else:
            # Fallback : charger les données pour obtenir les noms
            data_path = r"C:\Users\Zouhire\Documents\GitHub\PROJET___-2__-GROUPE__7__CODING__WEEK-1\data_clean.csv"
            df = pd.read_csv(data_path)
            target = "NObeyesdad"  # à adapter selon votre dataset
            feature_names = [c for c in df.columns if c != target]
        # Créer une ligne de test (valeurs nulles)
        sample = pd.DataFrame([[0]*len(feature_names)], columns=feature_names)
        pred = model.predict(sample)
        assert pred is not None
        assert len(pred) == 1