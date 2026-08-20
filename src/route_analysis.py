import requests
import math


# ============================================================
# SafeRoute AI - Route Analysis
# ============================================================


def get_coordinates(place):

    """
    Convert a place name into latitude/longitude
    using OpenStreetMap Nominatim.
    """

    url = "https://nominatim.openstreetmap.org/search"

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

    data = response.json()

    if not data:
        return None

    latitude = float(data[0]["lat"])
    longitude = float(data[0]["lon"])

    return latitude, longitude


def get_route(start, end):

    """
    Get driving route using OSRM.
    """

    start_lat, start_lon = start
    end_lat, end_lon = end

    url = (
        "https://router.project-osrm.org/route/v1/driving/"
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

    if data["code"] != "Ok":
        return None

    route = data["routes"][0]

    coordinates = route["geometry"]["coordinates"]

    # Convert longitude, latitude
    # into latitude, longitude
    route_coordinates = [
        [lat, lon]
        for lon, lat in coordinates
    ]

    distance_km = route["distance"] / 1000

    duration_minutes = route["duration"] / 60

    return {
        "coordinates": route_coordinates,
        "distance_km": distance_km,
        "duration_minutes": duration_minutes
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    source = "Hyderabad, India"

    destination = "Vijayawada, India"

    print("\nFinding source...")

    start = get_coordinates(source)

    print("Source:", start)

    print("\nFinding destination...")

    end = get_coordinates(destination)

    print("Destination:", end)

    if start and end:

        print("\nCalculating route...")

        route = get_route(
            start,
            end
        )

        if route:

            print("\n==============================")
            print("ROUTE FOUND")
            print("==============================")

            print(
                f"Distance: "
                f"{route['distance_km']:.2f} km"
            )

            print(
                f"Duration: "
                f"{route['duration_minutes']:.1f} minutes"
            )

            print(
                f"Route points: "
                f"{len(route['coordinates'])}"
            )

        else:

            print("Could not calculate route.")

    else:

        print("Could not find locations.")