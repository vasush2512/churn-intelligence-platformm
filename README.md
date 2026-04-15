# Customer Churn Intelligence Platform

An end-to-end churn prediction project with:
- Flask backend API
- Plotly Dash frontend
- Trained ML artifacts in `models/`

## Project Structure

```text
churn_project/
  api/
  dashboard/
  data/
  ml/
  models/
  requirements.txt
  run_project.py
```

## Quick Start

1. Activate environment:

```powershell
& "C:\Users\Guru\OneDrive\Desktop\churn_project\.venv310\Scripts\Activate.ps1"
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run frontend + backend together:

```powershell
python run_project.py
```

## URLs

- Frontend (Dash): `http://127.0.0.1:8050`
- Backend API (Flask): `http://127.0.0.1:5000`
- API Health: `http://127.0.0.1:5000/health`

## Notes

- If model files are missing, run training first from `ml/`.
- Keep virtual environment folders (`.venv*`) out of git tracking.
