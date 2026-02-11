"""
Machine Learning Training Pipeline for Bus Delay Prediction
Trains gradient boosting model using XGBoost framework
Predicts arrival delays based on traffic, crowd, and weather factors
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import time

print("=" * 70)
print("  TRANSIT DELAY PREDICTION MODEL - TRAINING PIPELINE")
print("=" * 70)

# ===========================
# CONFIGURATION
# ===========================

TRAINING_DATA_FILE = "historical_bus_data.csv"
OUTPUT_MODEL_FILE = "bus_prediction_model.json"
RANDOM_SEED = 42

# Model hyperparameters (tuned for transit prediction)
MODEL_PARAMS = {
    "n_estimators": 120,
    "learning_rate": 0.08,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED
}

# ===========================
# DATA LOADING
# ===========================

print(f"\n[STEP 1] Loading training data...")

if not os.path.exists(TRAINING_DATA_FILE):
    print(f"[ERROR] Training data file not found: {TRAINING_DATA_FILE}")
    print(f"[INFO] Please run data generation script first")
    exit(1)

# Load historical data
training_dataframe = pd.read_csv(TRAINING_DATA_FILE)

print(f"[SUCCESS] Loaded {len(training_dataframe)} training examples")
print(f"\nDataset preview:")
print(training_dataframe.head(10))
print(f"\nDataset statistics:")
print(training_dataframe.describe())

# ===========================
# DATA PREPARATION
# ===========================

print(f"\n[STEP 2] Preparing features and target variable...")

# Define feature columns (input variables)
FEATURE_COLUMNS = ["traffic_delay", "crowd_count", "is_raining"]
TARGET_COLUMN = "actual_arrival_time"

# Verify all required columns exist
missing_columns = []
for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
    if col not in training_dataframe.columns:
        missing_columns.append(col)

if missing_columns:
    print(f"[ERROR] Missing required columns: {missing_columns}")
    exit(1)

# Extract features (X) and target (y)
X_features = training_dataframe[FEATURE_COLUMNS]
y_target = training_dataframe[TARGET_COLUMN]

print(f"[SUCCESS] Features shape: {X_features.shape}")
print(f"[SUCCESS] Target shape: {y_target.shape}")

# ===========================
# TRAIN-TEST SPLIT
# ===========================

print(f"\n[STEP 3] Splitting data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y_target,
    test_size=0.25,
    random_state=RANDOM_SEED,
    shuffle=True
)

print(f"[SUCCESS] Training set: {len(X_train)} samples")
print(f"[SUCCESS] Testing set: {len(X_test)} samples")
print(f"[INFO] Train/Test ratio: {len(X_train)/len(X_test):.2f}:1")

# ===========================
# MODEL TRAINING
# ===========================

print(f"\n[STEP 4] Training XGBoost regression model...")
print(f"[CONFIG] Model hyperparameters:")
for param, value in MODEL_PARAMS.items():
    print(f"         {param}: {value}")

start_time = time.time()

# Initialize XGBoost regressor
transit_model = XGBRegressor(**MODEL_PARAMS)

# Train on training data
transit_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

training_duration = time.time() - start_time

print(f"[SUCCESS] Model training completed in {training_duration:.2f} seconds")

# ===========================
# MODEL EVALUATION
# ===========================

print(f"\n[STEP 5] Evaluating model performance...")

# Generate predictions
train_predictions = transit_model.predict(X_train)
test_predictions = transit_model.predict(X_test)

# Calculate metrics for training set
train_mae = mean_absolute_error(y_train, train_predictions)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
train_r2 = r2_score(y_train, train_predictions)

# Calculate metrics for testing set
test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_r2 = r2_score(y_test, test_predictions)

print(f"\n{'Metric':<25} | {'Training':<12} | {'Testing':<12}")
print("-" * 55)
print(f"{'Mean Absolute Error':<25} | {train_mae:<12.3f} | {test_mae:<12.3f}")
print(f"{'Root Mean Squared Error':<25} | {train_rmse:<12.3f} | {test_rmse:<12.3f}")
print(f"{'R² Score':<25} | {train_r2:<12.3f} | {test_r2:<12.3f}")

# ===========================
# CROSS-VALIDATION
# ===========================

print(f"\n[STEP 6] Running cross-validation (5-fold)...")

cv_scores = cross_val_score(
    transit_model,
    X_features,
    y_target,
    cv=5,
    scoring='neg_mean_absolute_error'
)

cv_mae_scores = -cv_scores  # Convert to positive MAE
print(f"[SUCCESS] Cross-validation MAE scores: {cv_mae_scores}")
print(f"[SUCCESS] Average CV MAE: {cv_mae_scores.mean():.3f} (+/- {cv_mae_scores.std():.3f})")

# ===========================
# FEATURE IMPORTANCE
# ===========================

print(f"\n[STEP 7] Analyzing feature importance...")

feature_importance = transit_model.feature_importances_
importance_ranking = sorted(
    zip(FEATURE_COLUMNS, feature_importance),
    key=lambda x: x[1],
    reverse=True
)

print(f"\nFeature importance ranking:")
for rank, (feature, importance) in enumerate(importance_ranking, 1):
    print(f"  {rank}. {feature:<20} : {importance:.4f}")

# ===========================
# PREDICTION EXAMPLES
# ===========================

print(f"\n[STEP 8] Testing with sample scenarios...")

test_scenarios = [
    {"traffic_delay": 5.0, "crowd_count": 10, "is_raining": 0, "description": "Light traffic, moderate crowd, clear weather"},
    {"traffic_delay": 15.0, "crowd_count": 25, "is_raining": 1, "description": "Heavy traffic, large crowd, raining"},
    {"traffic_delay": 0.0, "crowd_count": 2, "is_raining": 0, "description": "No traffic, small crowd, clear weather"},
    {"traffic_delay": 8.0, "crowd_count": 0, "is_raining": 0, "description": "Moderate traffic, no crowd, clear weather"}
]

print(f"\nSample predictions:")
for scenario in test_scenarios:
    input_features = pd.DataFrame([{
        "traffic_delay": scenario["traffic_delay"],
        "crowd_count": scenario["crowd_count"],
        "is_raining": scenario["is_raining"]
    }])
    
    prediction = transit_model.predict(input_features)[0]
    
    print(f"\n  Scenario: {scenario['description']}")
    print(f"  Traffic: {scenario['traffic_delay']:.1f}min | Crowd: {scenario['crowd_count']} | Rain: {'Yes' if scenario['is_raining'] else 'No'}")
    print(f"  → Predicted Delay: {prediction:.2f} minutes")

# ===========================
# MODEL PERSISTENCE
# ===========================

print(f"\n[STEP 9] Saving trained model...")

transit_model.save_model(OUTPUT_MODEL_FILE)

print(f"[SUCCESS] Model saved to: {OUTPUT_MODEL_FILE}")
print(f"[INFO] Model file size: {os.path.getsize(OUTPUT_MODEL_FILE) / 1024:.2f} KB")

# ===========================
# TRAINING SUMMARY
# ===========================

print(f"\n" + "=" * 70)
print("  TRAINING SUMMARY")
print("=" * 70)
print(f"  Training Samples: {len(X_train)}")
print(f"  Testing Samples: {len(X_test)}")
print(f"  Test MAE: {test_mae:.3f} minutes")
print(f"  Test R² Score: {test_r2:.3f}")
print(f"  Training Duration: {training_duration:.2f} seconds")
print(f"  Output Model: {OUTPUT_MODEL_FILE}")
print(f"  Status: {'✓ READY FOR DEPLOYMENT' if test_mae < 3.0 else '⚠ NEEDS TUNING'}")
print("=" * 70)

print(f"\n[COMPLETE] Training pipeline finished successfully!")
