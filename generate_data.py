"""
generate_data.py
================
Generates synthetic Formula 1 lap data for training the Lap Time Predictor.

Features generated:
- avg_speed_kph     : Average speed during the lap (km/h)
- tire_type         : Compound used (Soft=0, Medium=1, Hard=2, Inter=3, Wet=4)
- sector1_time      : Sector 1 time in seconds
- sector2_time      : Sector 2 time in seconds
- sector3_time      : Sector 3 time in seconds
- fuel_load_kg      : Remaining fuel (kg) – heavier = slower
- track_temp_c      : Track surface temperature (°C)
- air_temp_c        : Ambient air temperature (°C)
- drs_activations   : Number of DRS zones activated per lap
- lap_number        : Lap number in the race (tire degrades over laps)

Target:
- lap_time_s        : Total lap time in seconds
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

TIRE_BASE_DELTA = {
    0: -0.8,   # Soft   – fastest but degrades most
    1:  0.0,   # Medium – baseline
    2:  0.6,   # Hard   – slower but durable
    3:  2.5,   # Intermediate
    4:  4.5,   # Wet    – slowest
}

TIRE_DEGRADATION_RATE = {
    0: 0.065,  # Soft   – loses ~0.065s per lap
    1: 0.030,  # Medium
    2: 0.015,  # Hard
    3: 0.020,  # Inter
    4: 0.010,  # Wet
}

N_SAMPLES = 3000

def generate_dataset(n=N_SAMPLES):
    tire_type     = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.30, 0.35, 0.20, 0.08, 0.07])
    lap_number    = np.random.randint(1, 60, size=n)
    fuel_load_kg  = np.maximum(2, 105 - lap_number * 1.7 + np.random.normal(0, 3, n))
    track_temp_c  = np.random.uniform(18, 55, n)
    air_temp_c    = track_temp_c - np.random.uniform(5, 20, n)
    drs_zones     = np.random.choice([1, 2, 3], size=n, p=[0.2, 0.55, 0.25])

    # Base lap time (seconds) around a 90-second average circuit
    base_time = 90.0

    # Speed effect: 180–340 kph range on track
    avg_speed = np.random.uniform(185, 335, n)
    speed_effect = -0.04 * (avg_speed - 260)   # faster speed → shorter time

    # Tire compound effect
    tire_delta = np.array([TIRE_BASE_DELTA[t] for t in tire_type])
    tire_deg   = np.array([TIRE_DEGRADATION_RATE[t] for t in tire_type]) * lap_number

    # Fuel effect: 0.03s per kg above 0
    fuel_effect = 0.03 * fuel_load_kg

    # Temperature optimum around 40°C track; deviation increases time
    temp_effect = 0.012 * np.abs(track_temp_c - 40)

    # DRS saves ~0.4s per zone
    drs_effect = -0.40 * drs_zones

    noise = np.random.normal(0, 0.25, n)

    lap_time = (
        base_time
        + speed_effect
        + tire_delta
        + tire_deg
        + fuel_effect
        + temp_effect
        + drs_effect
        + noise
    )

    # Derive sector times that sum to lap_time ± tiny noise
    s1_frac = np.random.uniform(0.27, 0.35, n)
    s2_frac = np.random.uniform(0.33, 0.40, n)
    s3_frac = 1.0 - s1_frac - s2_frac

    sector1 = lap_time * s1_frac + np.random.normal(0, 0.05, n)
    sector2 = lap_time * s2_frac + np.random.normal(0, 0.05, n)
    sector3 = lap_time * s3_frac + np.random.normal(0, 0.05, n)

    df = pd.DataFrame({
        "avg_speed_kph":  np.round(avg_speed, 2),
        "tire_type":      tire_type,
        "sector1_time":   np.round(sector1, 3),
        "sector2_time":   np.round(sector2, 3),
        "sector3_time":   np.round(sector3, 3),
        "fuel_load_kg":   np.round(fuel_load_kg, 2),
        "track_temp_c":   np.round(track_temp_c, 1),
        "air_temp_c":     np.round(air_temp_c, 1),
        "drs_activations": drs_zones,
        "lap_number":     lap_number,
        "lap_time_s":     np.round(lap_time, 3),
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lap_data.csv")
    df.to_csv(out, index=False)
    print(f"✅  Generated {len(df)} samples → {out}")
    print(df.describe().to_string())