"""
api/app.py
Flask REST API — serves churn predictions from the trained model.

Endpoints:
  POST /predict        → single customer prediction
  POST /predict/batch  → list of customers
  GET  /health         → status check
  GET  /model/info     → model metadata

Run:
  python api/app.py
  # OR with gunicorn (production):
  # gunicorn -w 4 -b 0.0.0.0:5000 api.app:app
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, abort

app = Flask(__name__)

# ── Load artifacts ─────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def _load(fname):
    with open(os.path.join(MODELS_DIR, fname), "rb") as f:
        return pickle.load(f)

try:
    MODEL        = _load("best_model.pkl")
    SCALER       = _load("scaler.pkl")
    ENCODERS     = _load("encoders.pkl")
    FEATURE_NAMES = _load("features.pkl")
    with open(os.path.join(MODELS_DIR, "meta.json")) as f:
        META = json.load(f)
    print(f"✅ Model loaded: {META['model_name']}  ROC-AUC: {META['roc_auc']}")
except FileNotFoundError:
    print("⚠️  No trained model found. Run `python ml/train.py` first.")
    MODEL = SCALER = ENCODERS = FEATURE_NAMES = META = None


# ── Preprocessing helpers ─────────────────────────────────────────────────
CAT_COLS = [
    "contract", "internet_service", "payment_method",
    "tech_support", "online_security", "paperless_billing",
    "dependents", "partner",
]

def preprocess_input(data: dict) -> np.ndarray:
    """Convert a raw input dict into the scaled feature vector."""
    df = pd.DataFrame([data])

    # Feature engineering
    df["charge_per_service"] = (
        df["monthly_charges"] / df["num_services"].replace(0, 1)
    ).round(2)
    df["high_value"] = (df["monthly_charges"] > 80).astype(int)
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"],
    )

    # Encode categoricals
    for col, le in ENCODERS.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError:
                df[col] = 0   # unseen label → default

    # Align columns to training feature order
    df = df.reindex(columns=FEATURE_NAMES, fill_value=0)

    return SCALER.transform(df)


def make_prediction(raw: dict) -> dict:
    X = preprocess_input(raw)
    prob  = float(MODEL.predict_proba(X)[0][1])
    label = int(prob >= 0.5)

    risk = "High" if prob >= 0.70 else ("Medium" if prob >= 0.40 else "Low")

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction":  label,
        "churn_label":       "Yes" if label else "No",
        "risk_level":        risk,
    }


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Customer Churn Prediction API",
        "endpoints": [
            "/health",
            "/model/info",
            "/predict",
            "/predict/batch",
        ],
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": MODEL is not None,
    })


@app.route("/model/info", methods=["GET"])
def model_info():
    if not META:
        abort(503, "Model not loaded")
    return jsonify(META)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single prediction.
    Body (JSON):
    {
      "tenure": 12,
      "monthly_charges": 79.5,
      "total_charges": 954.0,
      "num_services": 3,
      "senior_citizen": 0,
      "contract": "Month-to-month",
      "internet_service": "Fiber optic",
      "payment_method": "Electronic check",
      "tech_support": "No",
      "online_security": "No",
      "paperless_billing": "Yes",
      "dependents": "No",
      "partner": "No"
    }
    """
    if MODEL is None:
        abort(503, "Model not loaded. Run training first.")

    data = request.get_json(force=True, silent=True)
    if not data:
        abort(400, "Invalid or missing JSON body.")

    required = ["tenure", "monthly_charges", "total_charges", "num_services",
                "senior_citizen", "contract", "internet_service", "payment_method",
                "tech_support", "online_security", "paperless_billing", "dependents", "partner"]
    missing = [k for k in required if k not in data]
    if missing:
        abort(400, f"Missing fields: {missing}")

    result = make_prediction(data)
    return jsonify({"input": data, "prediction": result})


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction.
    Body: { "customers": [ {...}, {...} ] }
    """
    if MODEL is None:
        abort(503, "Model not loaded. Run training first.")

    body = request.get_json(force=True, silent=True)
    if not body or "customers" not in body:
        abort(400, "Expected JSON with key 'customers' (list).")

    customers = body["customers"]
    if not isinstance(customers, list) or len(customers) == 0:
        abort(400, "'customers' must be a non-empty list.")

    results = []
    for i, cust in enumerate(customers):
        try:
            pred = make_prediction(cust)
            results.append({"index": i, "prediction": pred, "error": None})
        except Exception as e:
            results.append({"index": i, "prediction": None, "error": str(e)})

    churn_count = sum(1 for r in results if r["prediction"] and r["prediction"]["churn_prediction"] == 1)

    return jsonify({
        "total":        len(results),
        "churn_count":  churn_count,
        "churn_rate":   round(churn_count / len(results), 4),
        "results":      results,
    })


# ── Error handlers ────────────────────────────────────────────────────────

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad Request", "message": str(e)}), 400

@app.errorhandler(503)
def unavailable(e):
    return jsonify({"error": "Service Unavailable", "message": str(e)}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
