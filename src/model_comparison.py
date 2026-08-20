import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# SafeRoute AI - Model Comparison
# ============================================================

DATA_FILE = "data/cleaned_accidents.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("              SAFEROUTE AI")
print("             MODEL COMPARISON")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# FEATURES
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

target = "risk_score"

X = df[features]
y = df[target]


# ============================================================
# FEATURE TYPES
# ============================================================

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


# ============================================================
# PREPROCESSOR
# ============================================================

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


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# TRAIN MODELS
# ============================================================

results = []

trained_pipelines = {}


for name, model in models.items():

    print("\n" + "-" * 70)
    print(f"Training: {name}")
    print("-" * 70)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    predictions = predictions.clip(
        0,
        1
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append(
        {
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    )

    trained_pipelines[name] = pipeline

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

print("\n" + "=" * 70)
print("                 MODEL RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
            "R2": "{:.4f}".format
        }
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_pipeline = trained_pipelines[
    best_model_name
]

print("\n" + "=" * 70)
print("                 BEST MODEL")
print("=" * 70)

print(
    f"\nSelected model: {best_model_name}"
)

print(
    f"R²: "
    f"{results_df.iloc[0]['R2']:.4f}"
)

print(
    f"MAE: "
    f"{results_df.iloc[0]['MAE']:.4f}"
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

Path("models").mkdir(
    exist_ok=True
)

joblib.dump(
    best_pipeline,
    "models/best_saferoute_model.joblib"
)

print(
    "\nBest model saved to:"
)

print(
    "models/best_saferoute_model.joblib"
)


# ============================================================
# SAVE RESULTS
# ============================================================

Path("data").mkdir(
    exist_ok=True
)

results_df.to_csv(
    "data/model_comparison.csv",
    index=False
)

print(
    "\nComparison saved to:"
)

print(
    "data/model_comparison.csv"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("              FEATURE IMPORTANCE")
print("=" * 70)


# Get preprocessing object
fitted_preprocessor = best_pipeline[
    "preprocessor"
]

trained_model = best_pipeline[
    "model"
]


# Get transformed feature names
feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


# Check whether model supports feature importance

if hasattr(
    trained_model,
    "feature_importances_"
):

    importances = (
        trained_model
        .feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances
        }
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print(
        "\nTop 20 features:\n"
    )

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    importance_df.to_csv(
        "data/feature_importance.csv",
        index=False
    )

    print(
        "\nFeature importance saved to:"
    )

    print(
        "data/feature_importance.csv"
    )

else:

    print(
        "Feature importance is not available "
        "for this model."
    )


print("\n" + "=" * 70)
print("             COMPARISON COMPLETED")
print("=" * 70)