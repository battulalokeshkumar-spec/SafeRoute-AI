# 🚗 SafeRoute AI

## AI-Powered Accident Risk Prediction & Route Safety System

SafeRoute AI is an AI-based road safety system that estimates accident risk using historical accident data, road conditions, environmental conditions, traffic density, and travel-time information.

The system also analyzes a selected driving route and identifies historical accident locations near that route.

---

## 🎯 Project Objective

The objective of SafeRoute AI is to provide a data-driven road safety analysis system that can:

- Predict road accident risk
- Analyze road and environmental conditions
- Identify historical accident hotspots
- Analyze a driving route
- Estimate route-level safety risk
- Display accident locations on an interactive map
- Visualize accident concentration using a heatmap

---

## 🧠 Machine Learning

The project uses supervised machine learning for accident risk estimation.

### Models Evaluated

Three regression models were compared:

1. Random Forest
2. Gradient Boosting
3. Extra Trees

### Final Model

**Gradient Boosting Regressor**

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Gradient Boosting** | **0.0517** | **0.0720** | **0.8884** |
| Random Forest | 0.0554 | 0.0749 | 0.8794 |
| Extra Trees | 0.0552 | 0.0788 | 0.8665 |

Gradient Boosting was selected because it achieved the best R² score and lowest MAE among the evaluated models.

---

## 📊 Dataset

The project uses a dataset containing **20,000 accident records**.

Important attributes include:

- City
- State
- Latitude
- Longitude
- Date
- Time
- Hour
- Day of Week
- Weekend indicator
- Road Type
- Number of Lanes
- Traffic Signal
- Weather
- Visibility
- Temperature
- Traffic Density
- Accident Cause
- Accident Severity
- Vehicles Involved
- Casualties
- Peak Hour
- Risk Score

---

## 🔄 System Architecture

```text
                    ┌─────────────────────┐
                    │   Accident Dataset  │
                    │     20,000 Records  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Cleaning     │
                    │   & Preprocessing    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Machine Learning     │
                    │ Model Comparison     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Gradient Boosting    │
                    │    Risk Model        │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌──────────────────┐       ┌──────────────────┐
        │ Road Risk        │       │ Route Analysis   │
        │ Prediction       │       │ OSRM + OSM       │
        └────────┬─────────┘       └────────┬─────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Streamlit Web App │
                    └─────────────────────┘