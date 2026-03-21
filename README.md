# 📡 Customer Churn Prediction System

An end-to-end Machine Learning system to predict telecom customer churn — built with Python, Scikit-learn, XGBoost, Flask, and Plotly Dash.

---

## 🗂️ Project Structure

```
churn_project/
│
├── data/
│   ├── generate_data.py      ← Synthetic telecom dataset generator (~7k rows)
│   └── telecom_churn.csv     ← Generated after running generate_data.py
│
├── ml/
│   ├── preprocess.py         ← Cleaning, feature engineering, encoding, scaling
│   └── train.py              ← SMOTE + LR / RF / XGBoost training & evaluation
│
├── api/
│   └── app.py                ← Flask REST API (single + batch prediction)
│
├── dashboard/
│   └── dashboard.py          ← Plotly Dash analytics + live predictor
│
├── models/                   ← Auto-created after training
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── features.pkl
│   ├── meta.json
│   ├── roc_curves.png
│   └── feature_importance.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone / download the project
cd churn_project

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Run in Order

### Step 1 — Generate dataset
```bash
python data/generate_data.py
# Output: data/telecom_churn.csv  (~7000 rows)
```

### Step 2 — Train models
```bash
python ml/train.py
# Trains Logistic Regression, Random Forest, XGBoost with SMOTE
# Saves best model + artifacts to /models/
# Outputs ROC-AUC scores and classification reports
```

### Step 3 — Start Flask API
```bash
python api/app.py
# API running at http://localhost:5000
```

### Step 4 — Start Dash Dashboard
```bash
python dashboard/dashboard.py
# Dashboard at http://localhost:8050
```

---

## 📡 API Reference

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

### `GET /model/info`
```json
{ "model_name": "XGBoost", "roc_auc": 0.9521, "features": [...] }
```

### `POST /predict`
**Request:**
```json
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
```
**Response:**
```json
{
  "prediction": {
    "churn_probability": 0.8342,
    "churn_prediction": 1,
    "churn_label": "Yes",
    "risk_level": "High"
  }
}
```

### `POST /predict/batch`
```json
{
  "customers": [ { ...customer1... }, { ...customer2... } ]
}
```

---

## 🧠 ML Pipeline

| Step | Detail |
|------|--------|
| Data | 7,000 synthetic telecom records |
| Cleaning | Null handling, type coercion |
| Feature Engineering | charge_per_service, tenure_group, high_value flag |
| Encoding | LabelEncoder on 8 categorical columns |
| Scaling | StandardScaler on all features |
| Imbalance | SMOTE oversampling on training set |
| Models | Logistic Regression, Random Forest, XGBoost |
| Selection | Best ROC-AUC on held-out test set (80/20 split) |

---

## 📊 Dashboard Features

- **KPI Cards** — Total customers, churn count, avg charges, avg tenure
- **Churn by Contract** — Grouped bar chart
- **Monthly Charges Distribution** — Overlapping histogram (churn vs retain)
- **Churn by Internet Service** — Pie chart
- **Scatter Plot** — Tenure vs Charges coloured by churn
- **Live Predictor** — Adjust sliders and dropdowns → instant prediction

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| ML | Scikit-learn, XGBoost, imbalanced-learn |
| API | Flask |
| Dashboard | Plotly Dash, Dash Bootstrap Components |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Plotly |

---

## 📄 Interview Talking Points

- **Why SMOTE?** The churn dataset is class-imbalanced (~26% churn). SMOTE creates synthetic minority samples in feature space rather than simple duplication, preventing the model from just predicting "No Churn" for everything.
- **Why XGBoost over Random Forest?** XGBoost often outperforms on tabular data due to gradient boosting (sequential correction of errors) vs bagging. We validated this empirically via ROC-AUC on the test set.
- **Why ROC-AUC over Accuracy?** Accuracy is misleading on imbalanced datasets. ROC-AUC measures the model's ability to rank positive examples higher regardless of threshold.
- **How would you productionize this?** Containerize with Docker, serve Flask via Gunicorn + Nginx, add MLflow for experiment tracking, schedule retraining with Airflow or cron, monitor data drift with Evidently.
