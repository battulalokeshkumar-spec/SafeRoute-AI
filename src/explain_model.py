import pandas as pd
import joblib
import shap


# ============================================================
# SafeRoute AI - Model Explainability
# ============================================================

MODEL_PATH = "models/best_saferoute_model.joblib"
DATA_FILE = "data/cleaned_accidents.csv"


# Load model
pipeline = joblib.load(MODEL_PATH)

# Load data
df = pd.read_csv(DATA_FILE)


# ============================================================
# Features
# ============================================================

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


X = df[features]


# ============================================================
# Transform data
# ============================================================

preprocessor = pipeline["preprocessor"]

model = pipeline["model"]

X_transformed = preprocessor.transform(X)


# ============================================================
# Feature names
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


# ============================================================
# SHAP Explainer
# ============================================================

print("=" * 60)
print("       SAFEROUTE AI - SHAP EXPLAINABILITY")
print("=" * 60)

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(
    model
)


# Use a small sample first
sample = X_transformed[:100]

print("Calculating SHAP values...")

shap_values = explainer.shap_values(
    sample
)


# ============================================================
# Global Feature Importance
# ============================================================

importance = abs(shap_values).mean(
    axis=0
)

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Mean_SHAP": importance
    }
)

importance_df = importance_df.sort_values(
    "Mean_SHAP",
    ascending=False
)


print("\n" + "=" * 60)
print("TOP RISK FACTORS")
print("=" * 60)

print(
    importance_df.head(15).to_string(
        index=False
    )
)


# ============================================================
# Save results
# ============================================================

importance_df.to_csv(
    "data/shap_feature_importance.csv",
    index=False
)


print(
    "\nSHAP results saved to:"
)

print(
    "data/shap_feature_importance.csv"
)

print("\nExplainability analysis completed.")