"""
ml/train.py
Trains Logistic Regression, Random Forest, and XGBoost with SMOTE.
Saves best model + scaler + encoders to /models/.
Run: python ml/train.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    classification_report, roc_auc_score,
    confusion_matrix, RocCurveDisplay,
)
from imblearn.over_sampling  import SMOTE
from xgboost                 import XGBClassifier

warnings.filterwarnings("ignore")

# Local module
import sys; sys.path.insert(0, ".")
from ml.preprocess import preprocess

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Model definitions ─────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=0.5),
    "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=10,
                                                   random_state=42, n_jobs=-1),
    "XGBoost":              XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                          use_label_encoder=False, eval_metric="logloss",
                                          random_state=42, n_jobs=-1),
}


def apply_smote(X_train, y_train):
    print(f"\nBefore SMOTE → Churn: {y_train.sum()}  Non-churn: {(y_train==0).sum()}")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After  SMOTE → Churn: {y_res.sum()}  Non-churn: {(y_res==0).sum()}")
    return X_res, y_res


def evaluate(name, model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  ROC-AUC : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    cm = confusion_matrix(y_test, y_pred)
    return {"name": name, "model": model, "auc": auc, "y_proba": y_proba, "cm": cm}


def plot_roc_curves(results, y_test, save_path="models/roc_curves.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        RocCurveDisplay.from_predictions(
            y_test, r["y_proba"],
            name=f"{r['name']} (AUC={r['auc']:.3f})", ax=ax
        )
    ax.set_title("ROC Curves — Churn Prediction Models", fontsize=14, fontweight="bold")
    ax.plot([0,1],[0,1],"k--", linewidth=0.8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nROC curve saved → {save_path}")


def plot_feature_importance(model, feature_names, save_path="models/feature_importance.png"):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        return

    fi = pd.Series(imp, index=feature_names).sort_values(ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot(kind="barh", ax=ax, color="#2563EB", edgecolor="white")
    ax.set_title("Top Feature Importances (Best Model)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Feature importance chart saved → {save_path}")


def save_artifacts(best, scaler, encoders, feature_names):
    with open(f"{MODELS_DIR}/best_model.pkl", "wb")    as f: pickle.dump(best["model"],   f)
    with open(f"{MODELS_DIR}/scaler.pkl",     "wb")    as f: pickle.dump(scaler,           f)
    with open(f"{MODELS_DIR}/encoders.pkl",   "wb")    as f: pickle.dump(encoders,         f)
    with open(f"{MODELS_DIR}/features.pkl",   "wb")    as f: pickle.dump(feature_names,    f)

    meta = {
        "model_name": best["name"],
        "roc_auc":    round(best["auc"], 4),
        "features":   feature_names,
    }
    import json
    with open(f"{MODELS_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Artifacts saved to /{MODELS_DIR}/")
    print(f"   Best model  : {best['name']}  |  ROC-AUC: {best['auc']:.4f}")


def train():
    # 1. Preprocess
    X_train, X_test, y_train, y_test, scaler, encoders, feature_names = preprocess()

    # 2. SMOTE
    X_train_s, y_train_s = apply_smote(X_train, y_train)

    # 3. Train & evaluate all models
    results = []
    for name, model in MODELS.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_s, y_train_s)
        results.append(evaluate(name, model, X_test, y_test))

    # 4. Pick best by ROC-AUC
    best = max(results, key=lambda r: r["auc"])

    # 5. Plots
    plot_roc_curves(results, y_test)
    plot_feature_importance(best["model"], feature_names)

    # 6. Save artifacts
    save_artifacts(best, scaler, encoders, feature_names)

    return best, scaler, encoders, feature_names


if __name__ == "__main__":
    train()
