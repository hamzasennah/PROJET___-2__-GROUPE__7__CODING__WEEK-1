import pandas as pd
import numpy as np


# ---------------------------
# Charger les données
# ---------------------------
def load_data():

    df = pd.read_csv("data/data_clean.csv")

    return df


# ---------------------------
# Gestion valeurs manquantes
# ---------------------------
def handle_missing_values(df):

    df = df.fillna(df.mean(numeric_only=True))

    return df


# ---------------------------
# Optimisation mémoire
# ---------------------------
def optimize_memory(df):

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df