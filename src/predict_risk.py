import joblib
import pandas as pd

# Load trained model
MODEL_PATH = "models/saferoute_risk_model.joblib"

model = joblib.load(MODEL_PATH)


def predict_risk(
    city,
    state,
    latitude,
    longitude,
    hour,
    day_of_week,
    is_weekend,
    road_type,
    lanes,
    traffic_signal,
    weather,
    visibility,
    temperature,
    traffic_density,
    is_peak_hour
):
    data = pd.DataFrame([{
        "city": city,
        "state": state,
        "latitude": latitude,
        "longitude": longitude,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "road_type": road_type,
        "lanes": lanes,
        "traffic_signal": traffic_signal,
        "weather": weather,
        "visibility": visibility,
        "temperature": temperature,
        "traffic_density": traffic_density,
        "is_peak_hour": is_peak_hour
    }])

    prediction = model.predict(data)[0]

    # Keep score between 0 and 1
    prediction = max(0, min(1, prediction))

    percentage = prediction * 100

    if prediction < 0.35:
        level = "LOW"
    elif prediction < 0.65:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return prediction, percentage, level


# Test prediction
if __name__ == "__main__":

    score, percentage, level = predict_risk(
        city="Mumbai",
        state="Maharashtra",
        latitude=19.0760,
        longitude=72.8777,
        hour=18,
        day_of_week="Friday",
        is_weekend=0,
        road_type="Urban",
        lanes=4,
        traffic_signal=1,
        weather="Clear",
        visibility="Good",
        temperature=30,
        traffic_density="High",
        is_peak_hour=1
    )

    print("\n" + "=" * 50)
    print("        SAFEROUTE AI PREDICTION")
    print("=" * 50)

    print(f"\nRisk Score : {score:.2f}")
    print(f"Risk       : {percentage:.1f}%")
    print(f"Risk Level : {level}")