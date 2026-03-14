"""
backend.py  –  XAI-Lab inference backend
-----------------------------------------
A FastAPI server that:
  • Loads the model assigned to it via DATASET_ID env-var
    (model_1.joblib  or  model_2.joblib) from the shared PV.
  • Serves /predict  (POST) and /model-info (GET).
  • Reloads the model every 30 s so it picks up freshly
    trained artefacts without restarting the pod.
  • Handles SIGTERM gracefully and logs pod identity.

Environment variables:
    DATASET_ID      – "1" or "2"  (which model file to load)
    SHARED_VOLUME   – path to PV mount  (default /shared-volume)
"""

import os
import sys
import time
import signal
import socket
import threading
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_ID    = os.environ.get("DATASET_ID", "1")
SHARED_VOLUME = os.environ.get("SHARED_VOLUME", "/shared-volume")
MODEL_PATH    = os.path.join(SHARED_VOLUME, f"model_{DATASET_ID}.joblib")
RELOAD_SECS   = int(os.environ.get("MODEL_RELOAD_SECS", "30"))

# ── App & global state ─────────────────────────────────────────────────────────
app = FastAPI(title=f"Credit Risk API – Dataset {DATASET_ID}")

current_model    = None
feature_names    = None
last_trained_at  = None
model_metrics    = None


# ── Model loader ───────────────────────────────────────────────────────────────
def load_model():
    global current_model, feature_names, last_trained_at, model_metrics
    if not os.path.exists(MODEL_PATH):
        print(f"[{datetime.now()}] Model not found at {MODEL_PATH} – will retry.")
        return
    try:
        info             = joblib.load(MODEL_PATH)
        current_model    = info["model"]
        feature_names    = info["feature_names"]
        last_trained_at  = info["training_time"]
        model_metrics    = info.get("metrics", {})
        print(f"[{datetime.now()}] Loaded model from {MODEL_PATH} "
              f"(trained {last_trained_at})")
    except Exception as exc:
        print(f"[{datetime.now()}] Error loading model: {exc}")


def _periodic_reload(interval: int):
    while True:
        time.sleep(interval)
        load_model()


# ── SIGTERM handler ────────────────────────────────────────────────────────────
def _handle_sigterm(signum, frame):
    host = socket.gethostname()
    print(f"[{datetime.now()}] SIGTERM received on {host} "
          f"(DATASET_ID={DATASET_ID}, last trained={last_trained_at})")
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_sigterm)


# ── Input schema ──────────────────────────────────────────────────────────────
class CreditInput(BaseModel):
    Age:              int
    Sex:              str
    Job:              int
    Housing:          str
    Saving_accounts:  str
    Checking_account: float
    Credit_amount:    float
    Duration:         int
    Purpose:          str


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/model-info")
def model_info():
    if current_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {
        "status":          "active",
        "dataset_id":      DATASET_ID,
        "last_trained_at": last_trained_at,
        "metrics":         model_metrics,
        "feature_names":   feature_names,
        "model_type":      type(current_model).__name__,
        "host":            socket.gethostname(),
    }


@app.post("/predict")
def predict(data: CreditInput):
    if current_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Build DataFrame with training column names (spaces, not underscores)
    row = {
        "Age":              data.Age,
        "Sex":              data.Sex,
        "Job":              data.Job,
        "Housing":          data.Housing,
        "Saving accounts":  data.Saving_accounts,
        "Checking account": data.Checking_account,
        "Credit amount":    data.Credit_amount,
        "Duration":         data.Duration,
        "Purpose":          data.Purpose,
    }

    # Validate all expected features are present in the row
    missing = [f for f in feature_names if f not in row]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={"error": "Missing features", "missing": missing}
        )

    df   = pd.DataFrame([row])
    pred = current_model.predict(df)[0]
    prob = current_model.predict_proba(df)[0, 1]

    return {
        "Risk":             "Good" if pred == 1 else "Bad",
        "Probability":      float(prob),
        "dataset_id":       DATASET_ID,
        "model_trained_at": last_trained_at,
        "host":             socket.gethostname(),
    }


@app.get("/health")
def health():
    return {"status": "ok", "dataset_id": DATASET_ID,
            "model_loaded": current_model is not None}


# ── Startup ────────────────────────────────────────────────────────────────────
load_model()

_reload_thread = threading.Thread(
    target=_periodic_reload, args=(RELOAD_SECS,), daemon=True
)
_reload_thread.start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
