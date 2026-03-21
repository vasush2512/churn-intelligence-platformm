"""
ml/preprocess.py
Data cleaning, feature engineering, encoding, scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ── Categorical columns that need encoding ──────────────────────────────────
CAT_COLS = [
    "contract", "internet_service", "payment_method",
    "tech_support", "online_security", "paperless_billing",
    "dependents", "partner",
]

# ── Numeric columns ──────────────────────────────────────────────────────────
NUM_COLS = ["tenure", "monthly_charges", "total_charges", "num_services", "senior_citizen"]


def load_data(path: str = "data/telecom_churn.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop ID column — not a feature
    df.drop(columns=["customer_id"], inplace=True, errors="ignore")

    # Fix total_charges (can have blanks in real datasets)
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"].fillna(df["total_charges"].median(), inplace=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Avg monthly spend per service
    df["charge_per_service"] = (
        df["monthly_charges"] / df["num_services"].replace(0, 1)
    ).round(2)

    # Tenure buckets
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"],
    )

    # High-value flag
    df["high_value"] = (df["monthly_charges"] > 80).astype(int)

    return df


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = {}

    extra_cats = ["tenure_group"]   # engineered categoricals
    all_cats = CAT_COLS + extra_cats

    for col in all_cats:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    return df, encoders


def scale(X_train, X_test) -> tuple:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


def preprocess(path: str = "data/telecom_churn.csv", test_size: float = 0.2):
    """Full pipeline → returns X_train, X_test, y_train, y_test, scaler, encoders, feature_names."""
    df = load_data(path)
    df = clean(df)
    df = engineer_features(df)
    df, encoders = encode(df)

    target = "churn"
    X = df.drop(columns=[target])
    y = df[target]

    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    X_train_s, X_test_s, scaler = scale(X_train, X_test)

    print(f"Train: {X_train_s.shape}  |  Test: {X_test_s.shape}")
    print(f"Churn rate (train): {y_train.mean():.2%}  |  (test): {y_test.mean():.2%}")

    return X_train_s, X_test_s, y_train, y_test, scaler, encoders, feature_names


if __name__ == "__main__":
    preprocess()
