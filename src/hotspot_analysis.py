import pandas as pd
import math


# ============================================================
# SafeRoute AI - Route Hotspot Analysis
# ============================================================


ACCIDENT_FILE = "data/cleaned_accidents.csv"


# ============================================================
# Distance Calculation
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):

    """
    Calculate distance between two geographic coordinates
    using the Haversine formula.

    Returns distance in kilometers.
    """

    earth_radius = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

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
# Find Accidents Near Route
# ============================================================

def find_route_accidents(
    route_coordinates,
    accident_df,
    radius_km=1.0
):

    nearby_accidents = []

    for route_lat, route_lon in route_coordinates:

        # Rough geographic filtering first
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

        candidates = accident_df[
            (accident_df["latitude"]
             .between(
                 route_lat - lat_range,
                 route_lat + lat_range
             ))
            &
            (accident_df["longitude"]
             .between(
                 route_lon - lon_range,
                 route_lon + lon_range
             ))
        ]

        for _, accident in candidates.iterrows():

            distance = haversine_distance(
                route_lat,
                route_lon,
                accident["latitude"],
                accident["longitude"]
            )

            if distance <= radius_km:

                nearby_accidents.append(
                    {
                        "accident_id":
                            accident.get(
                                "accident_id",
                                None
                            ),

                        "latitude":
                            accident["latitude"],

                        "longitude":
                            accident["longitude"],

                        "distance_km":
                            distance,

                        "risk_score":
                            accident["risk_score"],

                        "severity":
                            accident["accident_severity"]
                    }
                )

    if not nearby_accidents:

        return pd.DataFrame()

    result = pd.DataFrame(
        nearby_accidents
    )

    # Remove duplicates because
    # multiple route points can detect
    # the same accident.
    result = result.drop_duplicates(
        subset=[
            "latitude",
            "longitude"
        ]
    )

    return result


# ============================================================
# Route Risk Calculation
# ============================================================

def calculate_route_risk(
    nearby_accidents,
    route_distance_km
):

    if nearby_accidents.empty:

        return {
            "risk_score": 0,
            "risk_percentage": 0,
            "risk_level": "LOW",
            "accidents_near_route": 0
        }

    accident_count = len(
        nearby_accidents
    )

    # Accident density per 10 km
    density = (
        accident_count
        /
        max(route_distance_km, 1)
    ) * 10

    # Average historical risk
    average_risk = nearby_accidents[
        "risk_score"
    ].mean()

    # Combine density and historical risk
    #
    # This is a project-level risk index,
    # not a real-world accident probability.

    density_component = min(
        density / 10,
        1
    )

    combined_score = (
        0.6 * average_risk
        +
        0.4 * density_component
    )

    combined_score = max(
        0,
        min(
            1,
            combined_score
        )
    )

    percentage = combined_score * 100

    if combined_score < 0.35:

        level = "LOW"

    elif combined_score < 0.65:

        level = "MEDIUM"

    else:

        level = "HIGH"

    return {
        "risk_score": combined_score,
        "risk_percentage": percentage,
        "risk_level": level,
        "accidents_near_route": accident_count
    }


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("       SAFEROUTE AI - HOTSPOT ANALYSIS")
    print("=" * 60)

    # Load accident dataset
    accident_df = pd.read_csv(
        ACCIDENT_FILE
    )

    print(
        f"\nLoaded {len(accident_df)} accident records."
    )

    # Import routing functions
    from route_analysis import (
        get_coordinates,
        get_route
    )

    source = "Hyderabad, India"
    destination = "Vijayawada, India"

    print("\nFinding locations...")

    start = get_coordinates(
        source
    )

    end = get_coordinates(
        destination
    )

    if not start or not end:

        print(
            "Could not find locations."
        )

        exit()

    print("\nCalculating route...")

    route = get_route(
        start,
        end
    )

    if not route:

        print(
            "Could not calculate route."
        )

        exit()

    print(
        f"Route distance: "
        f"{route['distance_km']:.2f} km"
    )

    print(
        f"Route points: "
        f"{len(route['coordinates'])}"
    )

    print(
        "\nSearching for accidents "
        "within 1 km of route..."
    )

    nearby = find_route_accidents(
        route["coordinates"],
        accident_df,
        radius_km=1.0
    )

    print(
        f"\nNearby accidents found: "
        f"{len(nearby)}"
    )

    # Calculate route risk

    result = calculate_route_risk(
        nearby,
        route["distance_km"]
    )

    print("\n" + "=" * 60)
    print("              ROUTE RISK")
    print("=" * 60)

    print(
        f"\nRisk Score: "
        f"{result['risk_score']:.2f}"
    )

    print(
        f"Risk Percentage: "
        f"{result['risk_percentage']:.1f}%"
    )

    print(
        f"Risk Level: "
        f"{result['risk_level']}"
    )

    print(
        f"Accidents Near Route: "
        f"{result['accidents_near_route']}"
    )