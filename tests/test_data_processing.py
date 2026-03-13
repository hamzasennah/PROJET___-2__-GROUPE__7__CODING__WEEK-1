import pandas as pd
import sys
import os

# permettre d'importer les fichiers du dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_processing import optimize_memory


def test_optimize_memory():
    # créer un petit dataframe test
    df = pd.DataFrame({
        "Age": [20, 30, 40],
        "Height": [1.70, 1.80, 1.65],
        "Weight": [70, 80, 60]
    })

    # mémoire avant
    before = df.memory_usage().sum()

    # appliquer la fonction
    df_optimized = optimize_memory(df)

    # mémoire après
    after = df_optimized.memory_usage().sum()

    # vérifier que la mémoire a diminué
    assert after <= before