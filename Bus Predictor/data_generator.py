"""
Synthetic Training Data Generator
Creates realistic historical bus delay data for ML training
Simulates various traffic, crowd, and weather conditions
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

print("=" * 70)
print("  SYNTHETIC BUS DELAY DATA GENERATOR")
print("=" * 70)

# ===========================
# CONFIGURATION
# ===========================

OUTPUT_FILENAME = "historical_bus_data.csv"
TOTAL_SAMPLES = 2000
RANDOM_SEED = 42

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ===========================
# SIMULATION PARAMETERS
# ===========================

# Traffic delay distribution (minutes)
TRAFFIC_CONFIG = {
    "min": 0.0,
    "max": 30.0,
    "peak_mean": 15.0,
    "peak_std": 5.0,
    "off_peak_mean": 3.0,
    "off_peak_std": 2.0
}

# Passenger crowd distribution
CROWD_CONFIG = {
    "min": 0,
    "max": 50,
    "rush_hour_mean": 25,
    "rush_hour_std": 8,
    "normal_mean": 8,
    "normal_std": 5
}

# Weather probability
WEATHER_CONFIG = {
    "rain_probability": 0.25,  # 25% chance of rain
}

# Delay calculation parameters
BOARDING_TIME_PER_PERSON = 4.5  # seconds per passenger
RAIN_DELAY_PENALTY = 2.5  # additional minutes when raining
NOISE_RANGE = (-1.5, 1.5)  # random variation in minutes

# ===========================
# DATA GENERATION
# ===========================

print(f"\n[GENERATING] Creating {TOTAL_SAMPLES} simulated bus delay records...")

generated_records = []

for sample_id in range(TOTAL_SAMPLES):
    # Determine if this is a peak hour sample (40% of samples)
    is_peak_hour = random.random() < 0.4
    
    # Generate traffic delay
    if is_peak_hour:
        traffic_delay = np.random.normal(
            TRAFFIC_CONFIG["peak_mean"],
            TRAFFIC_CONFIG["peak_std"]
        )
    else:
        traffic_delay = np.random.normal(
            TRAFFIC_CONFIG["off_peak_mean"],
            TRAFFIC_CONFIG["off_peak_std"]
        )
    
    # Clamp traffic delay to valid range
    traffic_delay = np.clip(
        traffic_delay,
        TRAFFIC_CONFIG["min"],
        TRAFFIC_CONFIG["max"]
    )
    traffic_delay = round(traffic_delay, 1)
    
    # Generate crowd count
    if is_peak_hour:
        crowd_count = int(np.random.normal(
            CROWD_CONFIG["rush_hour_mean"],
            CROWD_CONFIG["rush_hour_std"]
        ))
    else:
        crowd_count = int(np.random.normal(
            CROWD_CONFIG["normal_mean"],
            CROWD_CONFIG["normal_std"]
        ))
    
    # Clamp crowd to valid range
    crowd_count = np.clip(
        crowd_count,
        CROWD_CONFIG["min"],
        CROWD_CONFIG["max"]
    )
    
    # Generate weather condition
    is_raining = 1 if random.random() < WEATHER_CONFIG["rain_probability"] else 0
    
    # ===========================
    # CALCULATE ACTUAL DELAY
    # ===========================
    # This simulates the "ground truth" that the ML model will learn to predict
    
    # Base delay from traffic
    total_delay = traffic_delay
    
    # Add boarding time based on crowd size
    boarding_delay_seconds = crowd_count * BOARDING_TIME_PER_PERSON
    boarding_delay_minutes = boarding_delay_seconds / 60
    total_delay += boarding_delay_minutes
    
    # Add weather penalty
    if is_raining:
        total_delay += RAIN_DELAY_PENALTY
    
    # Add random noise to simulate real-world variation
    noise = random.uniform(NOISE_RANGE[0], NOISE_RANGE[1])
    total_delay += noise
    
    # Ensure delay is non-negative
    total_delay = max(0.5, total_delay)
    total_delay = round(total_delay, 2)
    
    # Store record
    generated_records.append({
        "traffic_delay": traffic_delay,
        "crowd_count": crowd_count,
        "is_raining": is_raining,
        "actual_arrival_time": total_delay
    })
    
    # Progress indicator
    if (sample_id + 1) % 500 == 0:
        print(f"[PROGRESS] Generated {sample_id + 1}/{TOTAL_SAMPLES} records...")

# ===========================
# CREATE DATAFRAME
# ===========================

print(f"\n[PROCESSING] Creating DataFrame...")

dataset = pd.DataFrame(generated_records)

# ===========================
# DATA QUALITY CHECKS
# ===========================

print(f"\n[VALIDATION] Running data quality checks...")

# Check for missing values
missing_values = dataset.isnull().sum()
if missing_values.any():
    print(f"[WARNING] Found missing values:")
    print(missing_values[missing_values > 0])
else:
    print(f"[✓] No missing values")

# Check data types
print(f"\n[INFO] Data types:")
print(dataset.dtypes)

# ===========================
# DATASET STATISTICS
# ===========================

print(f"\n[STATISTICS] Dataset overview:")
print(dataset.describe())

print(f"\n[DISTRIBUTION] Feature distributions:")
print(f"  Traffic Delay:")
print(f"    Mean: {dataset['traffic_delay'].mean():.2f} min")
print(f"    Std:  {dataset['traffic_delay'].std():.2f} min")
print(f"    Range: {dataset['traffic_delay'].min():.2f} - {dataset['traffic_delay'].max():.2f} min")

print(f"\n  Crowd Count:")
print(f"    Mean: {dataset['crowd_count'].mean():.2f} people")
print(f"    Std:  {dataset['crowd_count'].std():.2f} people")
print(f"    Range: {dataset['crowd_count'].min()} - {dataset['crowd_count'].max()} people")

print(f"\n  Weather:")
print(f"    Rainy samples: {dataset['is_raining'].sum()} ({dataset['is_raining'].mean()*100:.1f}%)")
print(f"    Clear samples: {(1-dataset['is_raining']).sum()} ({(1-dataset['is_raining'].mean())*100:.1f}%)")

print(f"\n  Actual Arrival Time (Target):")
print(f"    Mean: {dataset['actual_arrival_time'].mean():.2f} min")
print(f"    Std:  {dataset['actual_arrival_time'].std():.2f} min")
print(f"    Range: {dataset['actual_arrival_time'].min():.2f} - {dataset['actual_arrival_time'].max():.2f} min")

# ===========================
# SAMPLE PREVIEW
# ===========================

print(f"\n[PREVIEW] First 10 records:")
print(dataset.head(10))

print(f"\n[PREVIEW] Random 5 records:")
print(dataset.sample(5))

# ===========================
# SAVE TO FILE
# ===========================

print(f"\n[SAVING] Writing data to {OUTPUT_FILENAME}...")

dataset.to_csv(OUTPUT_FILENAME, index=False)

file_size_kb = os.path.getsize(OUTPUT_FILENAME) / 1024

print(f"[✓] Successfully saved {len(dataset)} records")
print(f"[INFO] File size: {file_size_kb:.2f} KB")

# ===========================
# COMPLETION SUMMARY
# ===========================

print(f"\n" + "=" * 70)
print("  GENERATION COMPLETE")
print("=" * 70)
print(f"  Output File: {OUTPUT_FILENAME}")
print(f"  Total Records: {len(dataset)}")
print(f"  Features: {', '.join(dataset.columns[:-1])}")
print(f"  Target: {dataset.columns[-1]}")
print(f"  Status: ✓ READY FOR TRAINING")
print("=" * 70)

print(f"\n[NEXT STEP] Run model_trainer.py to train the ML model")

import os
