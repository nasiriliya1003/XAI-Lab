"""
model_trainer.py  –  XAI-Lab Kubernetes trainer
-------------------------------------------------
Trains an XGBoost credit-risk model on ONE of the two datasets
(selected via the DATASET_ID env-var) and saves it to the shared
PersistentVolume so the inference backend can hot-reload it.

Usage (local):
    DATASET_ID=1 python model_trainer.py
    DATASET_ID=2 python model_trainer.py

In Kubernetes two separate CronJobs call this script with
DATASET_ID=1 and DATASET_ID=2 respectively.
"""

import os
import joblib
import pandas as pd
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_ID = os.environ.get("DATASET_ID", "1")          # "1" or "2"
SHARED_VOLUME = os.environ.get("SHARED_VOLUME", "/shared-volume")

DATA_FILES = {
    "1": "data/credit_dataset1.csv",
    "2": "data/credit_dataset2.csv",
}

# Output path: model_1.joblib  or  model_2.joblib
MODEL_PATH = os.path.join(SHARED_VOLUME, f"model_{DATASET_ID}.joblib")

TARGET = "Risk"
CATEGORICAL_COLS = ["Sex", "Housing", "Saving accounts", "Purpose"]
NUMERICAL_COLS  = ["Age", "Job", "Checking account", "Credit amount", "Duration"]
RANDOM_STATE    = 42

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set([TARGET] + CATEGORICAL_COLS + NUMERICAL_COLS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def encode_target(y: pd.Series) -> pd.Series:
    return y.map({"Good": 1, "Bad": 0})


# ── Main training routine ──────────────────────────────────────────────────────
def train_model():
    print(f"[{datetime.now()}] ── Starting training  (DATASET_ID={DATASET_ID}) ──")

    if DATASET_ID not in DATA_FILES:
        raise ValueError(f"DATASET_ID must be '1' or '2', got '{DATASET_ID}'")

    data_path = DATA_FILES[DATASET_ID]
    print(f"[{datetime.now()}] Loading data from {data_path}")
    df = load_data(data_path)
    print(f"[{datetime.now()}] Dataset shape: {df.shape}")

    # ── Feature / target split
    X = df.drop(columns=[TARGET])
    y = encode_target(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ── Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            ("num", "passthrough", NUMERICAL_COLS),
        ]
    )

    # ── Model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    print(f"[{datetime.now()}] Training model …")
    pipeline.fit(X_train, y_train)

    # ── Quick evaluation
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred),    4),
    }
    print(f"[{datetime.now()}] Test metrics: {metrics}")

    # ── Persist to shared volume
    os.makedirs(SHARED_VOLUME, exist_ok=True)
    model_info = {
        "model":         pipeline,
        "dataset_id":    DATASET_ID,
        "feature_names": CATEGORICAL_COLS + NUMERICAL_COLS,
        "training_time": datetime.now().isoformat(),
        "metrics":       metrics,
    }
    joblib.dump(model_info, MODEL_PATH)
    print(f"[{datetime.now()}] Model saved to {MODEL_PATH}")
    print(f"[{datetime.now()}] ── Training complete ──")
    return True


if __name__ == "__main__":
    train_model()
