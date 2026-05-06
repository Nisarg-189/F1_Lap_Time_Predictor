<div align="center">

```
███████╗ ██╗    ██╗      █████╗  ██████╗     ██████╗ ██████╗ ███████╗██████╗ 
██╔════╝███║    ██║     ██╔══██╗██╔══██╗    ██╔══██╗██╔══██╗██╔════╝██╔══██╗
█████╗  ╚██║    ██║     ███████║██████╔╝    ██████╔╝██████╔╝█████╗  ██║  ██║
██╔══╝   ██║    ██║     ██╔══██║██╔═══╝     ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║
██║      ██║    ███████╗██║  ██║██║         ██║      ██║  ██║███████╗██████╔╝
╚═╝      ╚═╝    ╚══════╝╚═╝  ╚═╝╚═╝         ╚═╝      ╚═╝  ╚═╝╚══════╝╚═════╝ 
```

**`PREDICTOR`** &nbsp;·&nbsp; **`v1.0`** &nbsp;·&nbsp; **`MACHINE LEARNING`** &nbsp;·&nbsp; **`FLASK`** &nbsp;·&nbsp; **`SCIKIT-LEARN`**

<br>

> *"In Formula 1, a tenth of a second is the difference between a champion and an also-ran.*
> *This model predicts where you land — before you cross the line."*

<br>

![Python](https://img.shields.io/badge/Python-3.9+-FFD600?style=for-the-badge&logo=python&logoColor=black)
![Flask](https://img.shields.io/badge/Flask-REST_API-E8002D?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Engine-FF6B35?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-RACE_READY-00E676?style=for-the-badge)

</div>

---

<br>

## ⚡ WHAT IS THIS?

A **full-stack machine learning app** that predicts Formula 1 lap times with surgical precision.

Feed it 10 real-world race parameters. Two trained models fight over the answer. The result lands on an F1-themed dashboard in milliseconds.

```
   YOU INPUT                    ML MAGIC                  YOU GET
─────────────────        ──────────────────────        ──────────────
  🟥 Tire Compound   ──►   Linear Regression      ──►  1:28.403
  💨 Avg Speed            +                            
  ⛽ Fuel Load        ──►   Random Forest (×200)   ──►  1:28.671
  🌡️ Track Temp           
  📍 Sector Splits        + Feature Importances        🟧 S1 | 🟨 S2 | ⬜ S3
  🔋 DRS Zones            + Accuracy Metrics            MAE · RMSE · R²
  🔢 Lap Number      
```

<br>

---

## 🗂️ FILE MAP

```
f1-lap-predictor/
│
├── 🧬  generate_data.py        ← Physics-based synthetic F1 lap generator
│                                  3,000 laps · 10 features · realistic noise
│
├── 🧠  train_model.py          ← Trains both models, exports metrics & weights
│                                  LR + RF · 5-fold CV · feature importance
│
├── 🌐  app.py                  ← Flask REST API (4 endpoints)
│                                  /predict · /metrics · /feature-importance
│
├── 🎨  index.html              ← Single-page F1 dashboard (zero dependencies)
│                                  Dark theme · Animated results · Live charts
│
├── 📋  requirements.txt        ← pip install -r requirements.txt
│
└── 📁  models/                 ← Auto-generated after training
    ├── linear_model.pkl
    ├── rf_model.pkl
    ├── scaler.pkl
    ├── metrics.json
    └── feature_importance.json
```

<br>

---

## 🚀 LAUNCH SEQUENCE

```bash
# ── STEP 1 ── Install
pip install -r requirements.txt

# ── STEP 2 ── Generate 3,000 synthetic F1 laps
python generate_data.py
#  ✅  Generated 3000 samples → lap_data.csv

# ── STEP 3 ── Train both models (takes ~10 seconds)
python train_model.py
#  📐 Training Linear Regression …   R²: 0.9985
#  🌲 Training Random Forest    …   R²: 0.9630
#  ✅  Models saved to ./models/

# ── STEP 4 ── Launch the app
python app.py
#  🌐  Open: http://127.0.0.1:5000
```

<br>

---

## 🧬 THE DATA ENGINE  `generate_data.py`

Real F1 telemetry is proprietary and NDA-locked. So we built a **physics-based simulator** that models the same cause-and-effect relationships real engineers use.

### How each feature shapes the lap time:

```
FEATURE              EFFECT ON LAP TIME                          MAGNITUDE
────────────────────────────────────────────────────────────────────────────

🚗 Avg Speed         Faster speed = shorter time                −0.04s / kph
                     (referenced against 260 kph baseline)       above 260

🟥 Soft Tire         −0.8s advantage vs Medium                  degrades
🟧 Medium Tire        0.0s  (baseline compound)                  0.030s/lap
⬜ Hard Tire          +0.6s  slower but doesn't degrade          0.015s/lap
🟩 Intermediate      +2.5s  wet-weather crossover                0.020s/lap
🟦 Wet Tire          +4.5s  full wet conditions                  0.010s/lap

⛽ Fuel Load          +0.03s per kg remaining                    up to +3.3s
                      (a full tank of 110kg = 3.3s slower)

🌡️ Track Temp         Optimum is 40°C. Deviation adds time       +0.012s/°C
                      (tyres outside thermal window)              off-peak

📡 DRS Zones          Each zone saves drag, gains speed          −0.40s/zone
                      (1, 2 or 3 zones per circuit)

🔢 Lap Number         Tire degrades with every lap               ×degradation
                      multiplied by compound-specific rate        rate above

📍 Sector Splits      Derived from lap time fractions            ±0.05s noise
                      S1: ~30%  ·  S2: ~36%  ·  S3: ~34%
```

**Result:** 3,000 rows that behave like real data — without leaking anyone's trade secrets.

<br>

---

## 🧠 THE MODEL BATTLE  `train_model.py`

### 〔 CONTENDER 1 〕 Linear Regression

```
How it works:   Fits a straight line (hyperplane) through all 10 dimensions
                lap_time = w₁·speed + w₂·tire + w₃·s1 + ... + b

Needs scaling?  YES — StandardScaler applied. Without it, avg_speed (260)
                drowns out tire_type (0–4) in the weight calculation.

Strength:       Fully interpretable. Every coefficient tells you exactly
                how much each feature moves the lap time.

Weakness:       Can't capture non-linear interactions. It doesn't know that
                a Soft tire on lap 40 is 2.6s slower than on lap 1.
```

### 〔 CONTENDER 2 〕 Random Forest ← 🏆 RECOMMENDED

```
How it works:   200 decision trees, each trained on a random data subset.
                Each tree votes on a prediction. Final answer = average.

                       Tree 1: 1:28.4
                       Tree 2: 1:28.7     →  Average  →  1:28.55
                         ...
                       Tree 200: 1:28.6

Needs scaling?  NO — tree splits are threshold-based, not magnitude-based.

Strength:       Captures non-linear relationships and feature interactions
                automatically. Tire compound × lap number = handled.

Weakness:       Black box. You can't easily explain *why* it picked a time
                (though feature_importances_ gives partial insight).
```

### The Evaluation Suite

Every model is put through this gauntlet before being saved:

```
METRIC    FORMULA                       WHAT IT PUNISHES
──────────────────────────────────────────────────────────────────
MAE       mean(|y_pred − y_true|)       All errors equally
RMSE      √mean((y_pred − y_true)²)    Large errors more severely
R²        1 − SS_res/SS_tot            Overall variance explained
5-fold CV  Train on 4 folds, test on 1  Unstable / overfit models
           Repeat 5 times, average R²
```

### Expected Results

```
┌─────────────────────┬──────────┬──────────┬────────┐
│ Model               │  MAE (s) │ RMSE (s) │   R²   │
├─────────────────────┼──────────┼──────────┼────────┤
│ Linear Regression   │  ~0.07   │  ~0.08   │ ~0.998 │
│ Random Forest       │  ~0.33   │  ~0.42   │ ~0.963 │
└─────────────────────┴──────────┴──────────┴────────┘

⚠️  Interesting result: LR beats RF here — because the synthetic data
    was generated with mostly LINEAR relationships. On real-world F1
    telemetry (chaotic, noisy, full of interactions), RF would dominate.
    This is an intentional lesson baked into the project.
```

<br>

---

## 🌐 THE API  `app.py`

```
┌──────────────────────────────────────────────────────────────────────┐
│  BASE URL:  http://127.0.0.1:5000                                    │
├────────────────────┬────────┬─────────────────────────────────────── │
│  ENDPOINT          │ METHOD │  RETURNS                                │
├────────────────────┼────────┼─────────────────────────────────────── │
│  /                 │  GET   │  Serves index.html                      │
│  /predict          │  POST  │  LR + RF predictions, sector breakdown  │
│  /metrics          │  GET   │  MAE, RMSE, R² for both models         │
│  /feature-         │  GET   │  RF feature importances (sorted)       │
│    importance      │        │                                         │
│  /health           │  GET   │  { "status": "ok", "models_loaded": T } │
└────────────────────┴────────┴─────────────────────────────────────── ┘
```

**Sample `/predict` payload:**

```json
POST /predict
{
  "avg_speed_kph": 285,
  "tire_type": 0,
  "sector1_time": 26.4,
  "sector2_time": 32.1,
  "sector3_time": 29.8,
  "fuel_load_kg": 45.0,
  "track_temp_c": 42.0,
  "air_temp_c": 28.0,
  "drs_activations": 2,
  "lap_number": 18
}
```

**Response:**

```json
{
  "random_forest":      { "lap_time_s": 88.271, "lap_time_fmt": "1:28.271" },
  "linear_regression":  { "lap_time_s": 88.403, "lap_time_fmt": "1:28.403" },
  "sector_breakdown":   { "s1_pct": 29.8, "s2_pct": 36.3, "s3_pct": 33.9 },
  "inputs_echo":        { "tire_label": "Soft", "..." : "..." }
}
```

<br>

---

## 🎓 THE LESSONS (READ THIS)

These are the four things this project is secretly teaching you:

```
  01  ──  ALWAYS BASELINE FIRST
          Never judge a complex model in isolation. Train LR first.
          If RF doesn't clearly beat it, your features are the problem,
          not your model choice.

  02  ──  SCALING IS NOT OPTIONAL FOR LR
          avg_speed lives at 260. tire_type lives at 0–4.
          Without StandardScaler, the weight optimizer is flying blind.
          RF doesn't care — splits are thresholds, not magnitudes.

  03  ──  CROSS-VALIDATION > A SINGLE TEST SPLIT
          One 80/20 split can be lucky. 5-fold CV runs the experiment
          5 times and averages — giving you an honest performance number.

  04  ──  FEATURE IMPORTANCE ≠ CAUSATION
          Sector times rank as highly important because they mathematically
          derive from lap time. They don't *cause* a fast lap — they *are*
          the lap. Real insights come from fuel load, tire, and speed.
```

<br>

---

## 🔭 WHAT'S NEXT

Level up this project in order of increasing difficulty:

```
  [ EASY ]     Replace synthetic data with real FastF1 telemetry
               pip install fastf1  →  actual 2023/2024 race data

  [ MEDIUM ]   Add XGBoost / LightGBM  →  likely beats RF on real data

  [ MEDIUM ]   Hyperparameter tuning with RandomizedSearchCV

  [ HARD ]     SHAP values for per-prediction explainability
               "Why did it predict 1:28.4 specifically?"

  [ HARD ]     Neural Network with PyTorch — compare DL vs classical ML

  [ EXPERT ]   Live telemetry ingestion during a race weekend
               Predict lap times in real-time as sector splits come in
```

<br>

---

<div align="center">

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Built as part of a self-directed ML journey.      │
│   Stack: Python · Flask · scikit-learn · HTML/CSS   │
│   Goal: Adelaide University — AI/ML · July 2027     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

*The fastest lap is the one you predicted correctly.*

</div>
