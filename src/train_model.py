import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# SafeRoute AI - First ML Model
# ==========================================

print("=" * 60)
print("        SAFEROUTE AI - RISK MODEL")
print("=" * 60)

# Load cleaned data
df = pd.read_csv("data/cleaned_accidents.csv")

print(f"\nDataset shape: {df.shape}")

# ------------------------------------------
# Features available BEFORE an accident
# ------------------------------------------

features = [
    "city",
    "state",
    "latitude",
    "longitude",
    "hour",
    "day_of_week",
    "is_weekend",
    "road_type",
    "lanes",
    "traffic_signal",
    "weather",
    "visibility",
    "temperature",
    "traffic_density",
    "is_peak_hour"
]

target = "risk_score"

X = df[features]
y = df[target]

print("\nFeatures:")
print(features)

print("\nTarget:")
print(target)

# ------------------------------------------
# Identify feature types
# ------------------------------------------

categorical_features = [
    "city",
    "state",
    "day_of_week",
    "road_type",
    "weather",
    "visibility",
    "traffic_density"
]

numerical_features = [
    "latitude",
    "longitude",
    "hour",
    "is_weekend",
    "lanes",
    "traffic_signal",
    "temperature",
    "is_peak_hour"
]

# ------------------------------------------
# Preprocessing
# ------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# ------------------------------------------
# Random Forest Model
# ------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ------------------------------------------
# Train/Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# ------------------------------------------
# Train
# ------------------------------------------

print("\nTraining Random Forest...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")

# ------------------------------------------
# Prediction
# ------------------------------------------

predictions = pipeline.predict(X_test)

# Keep predictions between 0 and 1
predictions = predictions.clip(0, 1)

# ------------------------------------------
# Evaluation
# ------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nMAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")

print("\nExample predictions:")

results = pd.DataFrame({
    "Actual Risk": y_test.values[:10],
    "Predicted Risk": predictions[:10]
})

print(results.to_string(index=False))

print("\nModel training completed successfully.")
# ==========================================
# Save trained model
# ==========================================

Path("models").mkdir(exist_ok=True)

MODEL_PATH = "models/saferoute_risk_model.joblib"

joblib.dump(pipeline, MODEL_PATH)

print("\nModel saved successfully!")
print(f"Location: {MODEL_PATH}")