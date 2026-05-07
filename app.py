"""
app.py
======
Flask REST API for the F1 Lap Time Predictor.

Endpoints:
  GET  /               → serves index.html
  POST /predict        → returns lap time prediction from both models
  GET  /metrics        → returns training metrics (MAE, RMSE, R²)
  GET  /feature-importance  → returns RF feature importances
  GET  /health         → simple health check

Run:
    python app.py
Then open  http://127.0.0.1:5000  in your browser.
"""

import json
import os
import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

app = Flask(__name__, template_folder=".")
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FEATURES  = [
    "avg_speed_kph",
    "tire_type",
    "sector1_time",
    "sector2_time",
    "sector3_time",
    "fuel_load_kg",
    "track_temp_c",
    "air_temp_c",
    "drs_activations",
    "lap_number",
]

# ── Load models at startup ──────────────────────────────────────────────────
def load_models():
    missing = []
    for name in ["linear_model.pkl", "rf_model.pkl", "scaler.pkl"]:
        if not os.path.exists(os.path.join(MODEL_DIR, name)):
            missing.append(name)
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}. "
            "Run  python generate_data.py  then  python train_model.py  first."
        )

    with open(f"{MODEL_DIR}/linear_model.pkl", "rb") as f:
        lr = pickle.load(f)
    with open(f"{MODEL_DIR}/rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return lr, rf, scaler


try:
    lr_model, rf_model, scaler = load_models()
    MODELS_LOADED = True
    print("✅  Models loaded successfully.")
except FileNotFoundError as e:
    print(f"⚠️   {e}")
    MODELS_LOADED = False


def seconds_to_laptime(s):
    """Convert float seconds → 'M:SS.mmm' string."""
    minutes = int(s) // 60
    seconds = s - minutes * 60
    return f"{minutes}:{seconds:06.3f}"


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": MODELS_LOADED})


@app.route("/metrics")
def metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(path):
        return jsonify({"error": "metrics.json not found. Train models first."}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/feature-importance")
def feature_importance():
    path = os.path.join(MODEL_DIR, "feature_importance.json")
    if not os.path.exists(path):
        return jsonify({"error": "feature_importance.json not found."}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/predict", methods=["POST"])
def predict():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded. Run training scripts first."}), 503

    data = request.get_json(force=True)

    # Validate
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        row = np.array([[float(data[f]) for f in FEATURES]])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400

    # Predictions
    row_scaled = scaler.transform(row)
    lr_pred    = float(lr_model.predict(row_scaled)[0])
    rf_pred    = float(rf_model.predict(row)[0])

    # Sector contributions (for the breakdown bar)
    sector_sum    = data["sector1_time"] + data["sector2_time"] + data["sector3_time"]
    s1_pct = round(data["sector1_time"] / sector_sum * 100, 1) if sector_sum else 33.3
    s2_pct = round(data["sector2_time"] / sector_sum * 100, 1) if sector_sum else 33.3
    s3_pct = round(100 - s1_pct - s2_pct, 1)

    # Tire label
    tire_labels = {0: "Soft", 1: "Medium", 2: "Hard", 3: "Intermediate", 4: "Wet"}
    tire_label  = tire_labels.get(int(data["tire_type"]), "Unknown")

    return jsonify({
        "linear_regression": {
            "lap_time_s":    round(lr_pred, 3),
            "lap_time_fmt":  seconds_to_laptime(lr_pred),
        },
        "random_forest": {
            "lap_time_s":    round(rf_pred, 3),
            "lap_time_fmt":  seconds_to_laptime(rf_pred),
        },
        "sector_breakdown": {
            "s1_pct": s1_pct,
            "s2_pct": s2_pct,
            "s3_pct": s3_pct,
        },
        "inputs_echo": {
            "tire_label": tire_label,
            **{k: data[k] for k in FEATURES},
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)