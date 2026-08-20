import streamlit as st
import pandas as pd
import joblib
import folium
import requests
import math

from streamlit_folium import st_folium
from folium.plugins import HeatMap


# ============================================================
# SAFEROUTE AI
# AI-Powered Accident Risk Prediction & Route Safety System
# ============================================================

MODEL_PATH = "models/best_saferoute_model.joblib"
ACCIDENT_FILE = "data/cleaned_accidents.csv"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SafeRoute AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_accident_data():
    return pd.read_csv(ACCIDENT_FILE)


model = load_model()
accident_df = load_accident_data()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .title {
        font-size: 46px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🚗 SafeRoute AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-Powered Accident Risk Prediction & Route Safety System
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🚗 SafeRoute AI")

    st.write(
        """
        SafeRoute AI analyzes historical accident
        patterns and road/environmental conditions
        to estimate road safety.
        """
    )

    st.divider()

    st.subheader("📊 Model Information")

    st.metric(
        "Training Records",
        "20,000"
    )

    st.metric(
        "Model",
        "Gradient Boosting"
    )

    st.metric(
        "R² Score",
        "0.8884"
    )

    st.metric(
        "MAE",
        "0.0517"
    )

    st.divider()

    st.caption(
        "AI-based road safety research project"
    )


# ============================================================
# ROAD RISK PREDICTION
# ============================================================

st.markdown(
    '<div class="section-title">🤖 AI Road Risk Prediction</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the conditions for the road you want to analyze."
)


col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    city = st.selectbox(
        "🏙️ City",
        [
            "Mumbai",
            "Pune",
            "Delhi",
            "Bengaluru",
            "Chennai",
            "Hyderabad",
            "Kolkata",
            "Chandigarh"
        ]
    )

    state = st.selectbox(
        "🗺️ State",
        [
            "Maharashtra",
            "Delhi",
            "Karnataka",
            "Tamil Nadu",
            "Telangana",
            "West Bengal",
            "Punjab"
        ]
    )

    road_type = st.selectbox(
        "🛣️ Road Type",
        [
            "Urban",
            "Highway",
            "Rural",
            "Expressway"
        ]
    )

    lanes = st.slider(
        "🛣️ Number of Lanes",
        1,
        8,
        4
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    traffic_density = st.selectbox(
        "🚦 Traffic Density",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    weather = st.selectbox(
        "🌦️ Weather",
        [
            "Clear",
            "Rain",
            "Fog",
            "Storm",
            "Cloudy"
        ]
    )

    visibility = st.selectbox(
        "👁️ Visibility",
        [
            "Good",
            "Moderate",
            "Poor"
        ]
    )

    temperature = st.slider(
        "🌡️ Temperature (°C)",
        0,
        50,
        30
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    hour = st.slider(
        "🕐 Hour of Travel",
        0,
        23,
        18
    )

    day_of_week = st.selectbox(
        "📅 Day",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    is_weekend = (
        1
        if day_of_week in ["Saturday", "Sunday"]
        else 0
    )

    is_peak_hour = (
        1
        if hour in [7, 8, 9, 17, 18, 19, 20]
        else 0
    )

    traffic_signal = st.selectbox(
        "🚥 Traffic Signal",
        [0, 1],
        format_func=lambda x:
        "Yes" if x == 1 else "No"
    )


# ============================================================
# CITY COORDINATES
# ============================================================

coordinates = {

    "Mumbai":
        (19.0760, 72.8777),

    "Pune":
        (18.5204, 73.8567),

    "Delhi":
        (28.6139, 77.2090),

    "Bengaluru":
        (12.9716, 77.5946),

    "Chennai":
        (13.0827, 80.2707),

    "Hyderabad":
        (17.3850, 78.4867),

    "Kolkata":
        (22.5726, 88.3639),

    "Chandigarh":
        (30.7333, 76.7794)
}


latitude, longitude = coordinates[city]


# ============================================================
# ANALYZE ROAD RISK
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 ANALYZE ROAD RISK",
    use_container_width=True,
    type="primary"
)


if predict_button:

    input_data = pd.DataFrame(
        [{
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
        }]
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    try:

        prediction = model.predict(
            input_data
        )[0]

        prediction = max(
            0,
            min(1, prediction)
        )

        percentage = prediction * 100


    except Exception as error:

        st.error(
            f"❌ Prediction failed: {error}"
        )

        st.stop()


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if prediction < 0.35:

        risk_level = "LOW"

        message = (
            "Road conditions appear relatively safe."
        )

    elif prediction < 0.65:

        risk_level = "MEDIUM"

        message = (
            "Drive carefully and remain alert."
        )

    else:

        risk_level = "HIGH"

        message = (
            "High-risk conditions detected. "
            "Drive with extra caution."
        )


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.header("📊 AI Risk Analysis")

    result1, result2, result3 = st.columns(3)

    with result1:

        st.metric(
            "Risk Score",
            f"{percentage:.1f}%"
        )

    with result2:

        st.metric(
            "Risk Level",
            risk_level
        )

    with result3:

        st.metric(
            "Travel Hour",
            f"{hour}:00"
        )


    if risk_level == "LOW":

        st.success(
            f"🟢 LOW RISK — {message}"
        )

    elif risk_level == "MEDIUM":

        st.warning(
            f"🟡 MEDIUM RISK — {message}"
        )

    else:

        st.error(
            f"🔴 HIGH RISK — {message}"
        )


    st.progress(
        float(prediction)
    )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.subheader(
        "📋 Analysis Details"
    )

    details = pd.DataFrame(
        {
            "Parameter": [
                "City",
                "State",
                "Road Type",
                "Traffic Density",
                "Weather",
                "Visibility",
                "Temperature",
                "Lanes",
                "Traffic Signal",
                "Day",
                "Peak Hour"
            ],

            "Value": [
                city,
                state,
                road_type,
                traffic_density,
                weather,
                visibility,
                f"{temperature} °C",
                lanes,
                "Yes"
                if traffic_signal
                else "No",
                day_of_week,
                "Yes"
                if is_peak_hour
                else "No"
            ]
        }
    )

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ROUTE PLANNER
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🛣️ Safe Route Planner</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter your starting location and destination."
)


# ============================================================
# SESSION STATE
# ============================================================

if "route_result" not in st.session_state:
    st.session_state.route_result = None


# ============================================================
# ROUTE INPUTS
# ============================================================

route_col1, route_col2 = st.columns(2)


with route_col1:

    source = st.text_input(
        "📍 Starting Location",
        value="Hyderabad, India",
        key="source_location"
    )


with route_col2:

    destination = st.text_input(
        "🏁 Destination",
        value="Vijayawada, India",
        key="destination_location"
    )


# ============================================================
# CLEAR OLD RESULT WHEN ROUTE CHANGES
# ============================================================

current_route = (
    source.strip(),
    destination.strip()
)


if (
    "last_route_input"
    not in st.session_state
):

    st.session_state.last_route_input = current_route


elif (
    st.session_state.last_route_input
    != current_route
):

    st.session_state.route_result = None

    st.session_state.last_route_input = current_route


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_route = st.button(
    "🗺️ ANALYZE SAFE ROUTE",
    use_container_width=True,
    type="primary"
)


# ============================================================
# GEOCODING
# ============================================================

def get_coordinates(place):

    url = (
        "https://nominatim.openstreetmap.org/search"
    )

    params = {
        "q": place,
        "format": "json",
        "limit": 1
    }

    headers = {
        "User-Agent": "SafeRouteAI/1.0"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=10
    )

    response.raise_for_status()

    results = response.json()

    if not results:
        return None

    return (
        float(results[0]["lat"]),
        float(results[0]["lon"])
    )


# ============================================================
# ROUTE GENERATION
# ============================================================

def get_route(start, end):

    start_lat, start_lon = start
    end_lat, end_lon = end

    url = (
        "https://router.project-osrm.org/"
        "route/v1/driving/"
        f"{start_lon},{start_lat};"
        f"{end_lon},{end_lat}"
    )

    params = {
        "overview": "full",
        "geometries": "geojson"
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if data.get("code") != "Ok":
        return None

    route = data["routes"][0]

    coordinates = route[
        "geometry"
    ]["coordinates"]

    route_coordinates = [
        [lat, lon]
        for lon, lat in coordinates
    ]

    return {
        "coordinates": route_coordinates,
        "distance_km": route["distance"] / 1000,
        "duration_minutes":
            route["duration"] / 60
    }


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    earth_radius = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    delta_lat = math.radians(
        lat2 - lat1
    )

    delta_lon = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return earth_radius * c


# ============================================================
# FIND ACCIDENTS NEAR ROUTE
# ============================================================

def find_route_accidents(
    route_coordinates,
    accident_data,
    radius_km=1.0
):

    nearby = []

    step = max(
        1,
        len(route_coordinates) // 500
    )

    sampled_route = route_coordinates[::step]

    for route_lat, route_lon in sampled_route:

        lat_range = radius_km / 111

        lon_range = (
            radius_km
            /
            (
                111
                * math.cos(
                    math.radians(route_lat)
                )
                + 0.000001
            )
        )

        candidates = accident_data[
            (
                accident_data["latitude"]
                .between(
                    route_lat - lat_range,
                    route_lat + lat_range
                )
            )
            &
            (
                accident_data["longitude"]
                .between(
                    route_lon - lon_range,
                    route_lon + lon_range
                )
            )
        ]

        for _, accident in candidates.iterrows():

            distance = haversine_distance(
                route_lat,
                route_lon,
                accident["latitude"],
                accident["longitude"]
            )

            if distance <= radius_km:

                nearby.append(
                    {
                        "latitude":
                            accident["latitude"],

                        "longitude":
                            accident["longitude"],

                        "risk_score":
                            accident["risk_score"],

                        "severity":
                            accident["accident_severity"],

                        "distance_km":
                            distance
                    }
                )

    if not nearby:
        return pd.DataFrame()

    result = pd.DataFrame(nearby)

    return result.drop_duplicates(
        subset=[
            "latitude",
            "longitude"
        ]
    )


# ============================================================
# ROUTE RISK CALCULATION
# ============================================================

def calculate_route_risk(
    nearby_accidents,
    route_distance
):

    if nearby_accidents.empty:

        return {
            "score": 0,
            "percentage": 0,
            "level": "LOW",
            "count": 0
        }

    accident_count = len(
        nearby_accidents
    )

    density = (
        accident_count
        /
        max(route_distance, 1)
    ) * 10

    average_risk = nearby_accidents[
        "risk_score"
    ].mean()

    density_component = min(
        density / 10,
        1
    )

    score = (
        0.6 * average_risk
        +
        0.4 * density_component
    )

    score = max(
        0,
        min(
            1,
            score
        )
    )

    percentage = score * 100

    if score < 0.35:
        level = "LOW"

    elif score < 0.65:
        level = "MEDIUM"

    else:
        level = "HIGH"

    return {
        "score": score,
        "percentage": percentage,
        "level": level,
        "count": accident_count
    }


# ============================================================
# RUN ROUTE ANALYSIS
# ============================================================

if analyze_route:

    with st.spinner(
        "🔎 Analyzing route safety..."
    ):

        try:

            start = get_coordinates(
                source
            )

            if not start:

                st.error(
                    "❌ Could not find starting location."
                )

                st.stop()


            end = get_coordinates(
                destination
            )

            if not end:

                st.error(
                    "❌ Could not find destination."
                )

                st.stop()


            route = get_route(
                start,
                end
            )

            if not route:

                st.error(
                    "❌ Could not generate route."
                )

                st.stop()


            nearby = find_route_accidents(
                route["coordinates"],
                accident_df,
                radius_km=1.0
            )


            risk = calculate_route_risk(
                nearby,
                route["distance_km"]
            )


            # ==================================================
            # SAVE RESULT IN SESSION STATE
            # ==================================================

            st.session_state.route_result = {

                "source": source,

                "destination": destination,

                "start": start,

                "end": end,

                "route": route,

                "nearby": nearby,

                "risk": risk
            }


        except Exception as error:

            st.error(
                f"❌ Route analysis failed: {error}"
            )


# ============================================================
# DISPLAY SAVED ROUTE RESULT
# ============================================================

if st.session_state.route_result is not None:

    result = st.session_state.route_result

    start = result["start"]

    end = result["end"]

    route = result["route"]

    nearby = result["nearby"]

    risk = result["risk"]


    # ========================================================
    # ROUTE SUMMARY
    # ========================================================

    st.divider()

    st.header(
        "📊 Route Safety Analysis"
    )


    result1, result2, result3, result4 = st.columns(4)


    with result1:

        st.metric(
            "📏 Distance",
            f"{route['distance_km']:.1f} km"
        )


    with result2:

        st.metric(
            "⏱️ Estimated Time",
            f"{route['duration_minutes']:.0f} min"
        )


    with result3:

        st.metric(
            "⚠️ Nearby Accidents",
            risk["count"]
        )


    with result4:

        st.metric(
            "📊 Route Risk",
            f"{risk['percentage']:.1f}%"
        )


    # ========================================================
    # RISK MESSAGE
    # ========================================================

    if risk["level"] == "LOW":

        st.success(
            "🟢 LOW ROUTE RISK — "
            "Historical accident density along "
            "this route is relatively low."
        )

    elif risk["level"] == "MEDIUM":

        st.warning(
            "🟡 MEDIUM ROUTE RISK — "
            "Stay alert around identified "
            "historical accident zones."
        )

    else:

        st.error(
            "🔴 HIGH ROUTE RISK — "
            "Multiple historical accident hotspots "
            "were detected along this route."
        )


    # ========================================================
    # ROUTE MAP
    # ========================================================

    st.subheader(
        "🗺️ Route & Accident Hotspots"
    )


    center_lat = (
        start[0] + end[0]
    ) / 2

    center_lon = (
        start[1] + end[1]
    ) / 2


    route_map = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=7,
        tiles="OpenStreetMap"
    )


    # --------------------------------------------------------
    # ROUTE LINE
    # --------------------------------------------------------

    folium.PolyLine(
        route["coordinates"],
        weight=6,
        tooltip="Selected Route"
    ).add_to(route_map)


    # --------------------------------------------------------
    # START MARKER
    # --------------------------------------------------------

    folium.Marker(
        start,
        popup=(
            f"📍 Start: "
            f"{result['source']}"
        ),
        tooltip="Starting Location"
    ).add_to(route_map)


    # --------------------------------------------------------
    # DESTINATION MARKER
    # --------------------------------------------------------

    folium.Marker(
        end,
        popup=(
            f"🏁 Destination: "
            f"{result['destination']}"
        ),
        tooltip="Destination"
    ).add_to(route_map)


    # --------------------------------------------------------
    # ACCIDENT MARKERS
    # --------------------------------------------------------

    for _, accident in nearby.iterrows():

        folium.CircleMarker(

            location=[
                accident["latitude"],
                accident["longitude"]
            ],

            radius=5,

            popup=(
                "⚠️ Historical Accident<br>"
                f"Risk Score: "
                f"{accident['risk_score']:.2f}<br>"
                f"Severity: "
                f"{accident['severity']}<br>"
                f"Distance from route: "
                f"{accident['distance_km']:.2f} km"
            ),

            tooltip="Historical Accident",

            fill=True

        ).add_to(route_map)


    # --------------------------------------------------------
    # DISPLAY MAP
    # --------------------------------------------------------

    st_folium(
        route_map,
        width=None,
        height=650
    )


    # ========================================================
    # ACCIDENT DETAILS
    # ========================================================

    if not nearby.empty:

        st.subheader(
            "⚠️ Historical Accident Zones"
        )


        display_data = nearby[
            [
                "latitude",
                "longitude",
                "risk_score",
                "severity",
                "distance_km"
            ]
        ].copy()


        display_data.columns = [

            "Latitude",

            "Longitude",

            "Historical Risk",

            "Severity",

            "Distance from Route (km)"
        ]


        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "✅ No historical accident points "
            "were found within 1 km of this route."
        )


# ============================================================
# HISTORICAL ACCIDENT HEATMAP
# ============================================================

st.divider()

st.header(
    "🔥 Historical Accident Heatmap"
)

st.write(
    "Explore the geographic concentration "
    "of historical accident records."
)


heat_map = folium.Map(
    location=[
        20.5937,
        78.9629
    ],
    zoom_start=5,
    tiles="OpenStreetMap"
)


heat_data = (
    accident_df[
        [
            "latitude",
            "longitude"
        ]
    ]
    .dropna()
    .values
    .tolist()
)


HeatMap(
    heat_data,
    radius=12,
    blur=18,
    min_opacity=0.3
).add_to(heat_map)


st_folium(
    heat_map,
    width=None,
    height=600
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.divider()

st.header(
    "📈 Dataset Overview"
)


stat1, stat2, stat3, stat4 = st.columns(4)


with stat1:

    st.metric(
        "Accident Records",
        f"{len(accident_df):,}"
    )


with stat2:

    st.metric(
        "Cities",
        accident_df["city"].nunique()
    )


with stat3:

    st.metric(
        "States",
        accident_df["state"].nunique()
    )


with stat4:

    st.metric(
        "Average Risk",
        f"{accident_df['risk_score'].mean() * 100:.1f}%"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SafeRoute AI | AI-based road risk estimation system"
)

st.caption(
    "⚠️ Research/demo system. Risk scores are "
    "data-driven safety indices and do not guarantee "
    "accident prediction."
)