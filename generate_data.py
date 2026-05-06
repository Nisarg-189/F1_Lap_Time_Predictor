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
- fuel_load_kg      : Remaining fuel (kg) - heavier = slower
- track_temp_c      : Track surface temperature (°C)
- air_temp_c        : Ambient air temperature (°C)
- drs_activations   : Number of DRS zones activated per lap
- lap_number        : Lap number in the race (tire degrades over laps)
 
Target:
- lap_time_s        : Total lap time in seconds
"""
import numpy as np
import pandas as pd

np.random.seed(42)


TIRE_BASE_DELTA = {
    0: -0.8, # Soft: Fastest but degrades quickly
    1: 0.0,  # Medium: baseline
    2: 0.6,  # Hard: Slower but durable
    3: 2.5,  # Intermediate
    4: 4.5   # Wet: Slowest but safest in rain
}

TIRE_DEGRADATION_RATE = {
    0: 0.065, # Soft: loses 0.065s per lap
    1: 0.030, # Medium
    2: 0.015, # Hard
    3: 0.020, # Inter
    4: 0.010  # Wet 
}

N_SAMPLES = 3000

def generate_dataset(n=N_SAMPLES):
    tire_type = np.random.choice([0, 1, 2, 3, 4], size = n, p=[0.30, 0.35, 0.20, 0.08, 0.07])
    lap_number = np.random.randint(1, 60, size = n)
    fuel_load_kg = np.maximum(0, 105 - lap_number *1.75 + np.random.normal(0, 3 , n))
    track_temp_c= np.random.uniform(18, 55, n)
    air_temp_c = track_temp_c - np.random.uniform(5, 20, n)
    drs_zones = np.random.choice([1, 2, 3], size = n, p = [0.2, 0.55, 0.25])


    # Base lap time (seconds) around a 90 second average circuit
    base_time = 90

    # Speed effect : 180-350 kmph range on track
    avg_speed = np.random.uniform(180, 350, n)
    speed_effect = -0.04 * (avg_speed - 260) # Faster speed -> shorter time

    # Tire compound effect
    tire_delta = np.array([TIRE_BASE_DELTA[t] for t in tire_type])
    tire_deg = np.array([TIRE_DEGRADATION_RATE[t] for t in tire_type]) * lap_number

    # Fuel effect
    fuel_effect = 0.03 * fuel_load_kg

    # Temperature optimum around 40°C track; deviation increases time
    temp_effect = 0.012 * np.abs(track_temp_c - 40)

    # DRS saves 0.4 sec per zone
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


    # Derive ector times that sum to total laptime +- tiny noise
    s1_frac = np.random.uniform(0.27, 0.35, n)
    s2_frac = np.random.uniform(0.33, 0.40, n)
    s3_frac = 1 - s1_frac - s2_frac

    sector1 = lap_time * s1_frac + np.random.normal(0, 0.05, n)
    sector2 = lap_time * s2_frac + np.random.normal(0, 0.05, n)
    sector3 = lap_time * s3_frac + np.random.normal(0, 0.05, n)

    df = pd.DataFrame({
        "avg_speed_kmph": np.round(avg_speed, 2),
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
    df.to_csv("lap_data.csv", index = False)
    print(f"✅  Generated {len(df)} samples → lap_data.csv")
    print(df.describe().to_string())
