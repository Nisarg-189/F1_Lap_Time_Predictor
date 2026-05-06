"""
train_model.py
==============
Trains two models on the lap data:
  1. Linear Regression  – fast baseline, interpretable coefficients
  2. Random Forest      – captures non-linear interactions, usually wins
 
Outputs:
  - models/linear_model.pkl
  - models/rf_model.pkl
  - models/scaler.pkl
  - models/feature_importance.json
  - models/metrics.json
 
Run:
    python generate_data.py   # creates lap_data.csv
    python train_model.py     # trains & saves both models
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


# ── paths ──────────────────────────────────────────────────────────────────
DATA_PATH = "lap_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "avg_speed_kmph",
    "tire_type",
    "sector1_time",
    "sector2_time",
    "sector3_time",
    "fuel_load_kg",
    "track_temp_c",
    "air_temp_c",
    "drs_activations",
    "lap_number"]

TARGET = "lap_time_s"


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"'{DATA_PATH}' not found. Run  python generate_data.py  first."
        )
    df = pd.read_csv("lap_data.csv")
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def evaluate(name, model, X_test, y_test, scaler = None):
    X_input = scaler.transform(X_test) if scaler else X_test
    preds = model.predict(X_input)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_absolute_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"  MAE: {mae:.4f} s")
    print(f"  RMSE: {rmse:.4f} s")
    print(f"  R²: {r2:.4f}")
    return {"mae: ": round(mae, 4), "rmse: ": round(rmse, 4), "r2: ": round(r2, 4)}


def train():
    print("🏎️  Loading data …")
    X, y =load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


    # ── Scaler (needed for Linear Regression) ──────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

     # ── 1. Linear Regression ───────────────────────────────────────────────
    print("\n📐 Training Linear Regression …")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    cv_lr = cross_val_score(lr, X_train_scaled, y_train, cv = 5, scoring = "r2")
    print(f"  5-fold CV R²: {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")
    
    lr_metrics = evaluate("Linear Regression", lr, X_test, y_test, scaler= scaler)


    # ── 2. Random Forest ───────────────────────────────────────────────────
    print("\n🌲 Training Random Forest …")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
 
    cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")
    print(f"  5-fold CV R²: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
 
    rf_metrics = evaluate("Random Forest", rf, X_test, y_test)
 
    # ── Feature importances (Random Forest) ────────────────────────────────
    importances = dict(
        zip(FEATURES, [round(v, 5) for v in rf.feature_importances_])
    )
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    print("\n🔍 Feature Importances (RF):")
    for feat, imp in sorted_imp.items():
        bar = "█" * int(imp * 100)
        print(f"  {feat:<20} {imp:.4f}  {bar}")
 
    # ── Save ───────────────────────────────────────────────────────────────
    with open(f"{MODEL_DIR}/linear_model.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open(f"{MODEL_DIR}/rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
 
    with open(f"{MODEL_DIR}/feature_importance.json", "w") as f:
        json.dump(sorted_imp, f, indent=2)
 
    metrics = {
        "linear_regression": lr_metrics,
        "random_forest":     rf_metrics,
        "features":          FEATURES,
    }
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
 
    print(f"\n✅  Models saved to  ./{MODEL_DIR}/")
    print("    linear_model.pkl | rf_model.pkl | scaler.pkl")
    print("    feature_importance.json | metrics.json")
 
 
if __name__ == "__main__":
    train()