"""
generate_data.py
Generates a realistic synthetic telecom customer churn dataset (~7000 records).
Run: python data/generate_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 7000

def generate_dataset(n=N):
    tenure          = np.random.randint(1, 72, n)
    monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
    total_charges   = np.round(monthly_charges * tenure + np.random.normal(0, 50, n), 2)
    total_charges   = np.clip(total_charges, 0, None)

    contract        = np.random.choice(["Month-to-month", "One year", "Two year"],
                                       n, p=[0.55, 0.25, 0.20])
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"],
                                        n, p=[0.34, 0.44, 0.22])
    payment_method  = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )
    tech_support    = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22])
    online_security = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.28, 0.50, 0.22])
    paperless_bill  = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    senior_citizen  = np.random.choice([0, 1], n, p=[0.84, 0.16])
    dependents      = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])
    partner         = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
    num_services    = np.random.randint(1, 8, n)

    # Churn logic — correlated with real-world factors
    churn_prob = (
        0.05
        + 0.30 * (contract == "Month-to-month")
        + 0.10 * (internet_service == "Fiber optic")
        + 0.08 * (payment_method == "Electronic check")
        + 0.07 * (tech_support == "No")
        + 0.06 * (online_security == "No")
        - 0.15 * (tenure > 36)
        - 0.10 * (tenure > 60)
        + 0.05 * (monthly_charges > 80)
        + 0.04 * senior_citizen
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.90)
    churn      = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id":      [f"CUST{str(i).zfill(5)}" for i in range(n)],
        "tenure":           tenure,
        "monthly_charges":  monthly_charges,
        "total_charges":    total_charges,
        "contract":         contract,
        "internet_service": internet_service,
        "payment_method":   payment_method,
        "tech_support":     tech_support,
        "online_security":  online_security,
        "paperless_billing":paperless_bill,
        "senior_citizen":   senior_citizen,
        "dependents":       dependents,
        "partner":          partner,
        "num_services":     num_services,
        "churn":            churn,
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out = "data/telecom_churn.csv"
    df.to_csv(out, index=False)
    print(f"Dataset saved → {out}")
    print(f"Shape: {df.shape}  |  Churn rate: {df['churn'].mean():.2%}")
