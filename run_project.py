"""
Simple one-file launcher for churn_project.

Starts both services:
  - Backend API (Flask): http://127.0.0.1:5000
  - Frontend (Dash):     http://127.0.0.1:8050
"""

import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
API_PATH = os.path.join(PROJECT_ROOT, "api", "app.py")
DASHBOARD_PATH = os.path.join(PROJECT_ROOT, "dashboard", "dashboard.py")


def _exists_or_warn(path: str, label: str) -> None:
    if not os.path.exists(path):
        print(f"[WARN] {label} not found: {path}")


def main() -> None:
    print("Starting churn_project services...")
    print("Backend  -> http://127.0.0.1:5000")
    print("Frontend -> http://127.0.0.1:8050")
    print("Press Ctrl+C to stop both.")

    _exists_or_warn(os.path.join(PROJECT_ROOT, "data", "telecom_churn.csv"), "Dataset")
    _exists_or_warn(os.path.join(PROJECT_ROOT, "models", "best_model.pkl"), "Trained model")

    python = sys.executable
    api_proc = subprocess.Popen([python, API_PATH], cwd=PROJECT_ROOT)
    dash_proc = subprocess.Popen([python, DASHBOARD_PATH], cwd=PROJECT_ROOT)

    threading.Timer(2.0, lambda: webbrowser.open("http://127.0.0.1:8050")).start()

    try:
        while True:
            if api_proc.poll() is not None:
                print("[ERROR] API process exited.")
                break
            if dash_proc.poll() is not None:
                print("[ERROR] Dashboard process exited.")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        for proc in (api_proc, dash_proc):
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGINT)
                except Exception:
                    proc.terminate()
        for proc in (api_proc, dash_proc):
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("Stopped.")


if __name__ == "__main__":
    main()

