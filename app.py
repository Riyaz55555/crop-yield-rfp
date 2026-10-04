from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import requests

app = Flask(__name__)

# -------------------------
# Load trained model
# -------------------------
model = pickle.load(open("model_real.pkl", "rb"))

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("crop_india.csv")
df_clean = df.dropna().copy()

states   = sorted(df["State"].unique())
crops    = sorted(df["Crop"].unique())
seasons  = sorted(df["Season"].unique())

# Pre-compute avg rainfall per state (fallback when API fails)
state_avg_rainfall = df.groupby("State")["Annual_Rainfall"].mean().to_dict()

# Encoders (same fit as training)
le_state  = LabelEncoder().fit(df["State"])
le_crop   = LabelEncoder().fit(df["Crop"])
le_season = LabelEncoder().fit(df["Season"])

# -------------------------
# State → capital city map
# -------------------------
STATE_CAPITALS = {
    "Andhra Pradesh":            "Amaravati",
    "Arunachal Pradesh":         "Itanagar",
    "Assam":                     "Guwahati",
    "Bihar":                     "Patna",
    "Chhattisgarh":              "Raipur",
    "Delhi":                     "New Delhi",
    "Goa":                       "Panaji",
    "Gujarat":                   "Gandhinagar",
    "Haryana":                   "Chandigarh",
    "Himachal Pradesh":          "Shimla",
    "Jammu and Kashmir":         "Srinagar",
    "Jharkhand":                 "Ranchi",
    "Karnataka":                 "Bengaluru",
    "Kerala":                    "Thiruvananthapuram",
    "Madhya Pradesh":            "Bhopal",
    "Maharashtra":               "Mumbai",
    "Manipur":                   "Imphal",
    "Meghalaya":                 "Shillong",
    "Mizoram":                   "Aizawl",
    "Nagaland":                  "Kohima",
    "Odisha":                    "Bhubaneswar",
    "Punjab":                    "Chandigarh",
    "Rajasthan":                 "Jaipur",
    "Sikkim":                    "Gangtok",
    "Tamil Nadu":                "Chennai",
    "Telangana":                 "Hyderabad",
    "Tripura":                   "Agartala",
    "Uttar Pradesh":             "Lucknow",
    "Uttarakhand":               "Dehradun",
    "West Bengal":               "Kolkata",
    "Andaman and Nicobar Islands": "Port Blair",
    "Chandigarh":                "Chandigarh",
    "Dadra and Nagar Haveli":    "Silvassa",
    "Daman and Diu":             "Daman",
    "Lakshadweep":               "Kavaratti",
    "Puducherry":                "Puducherry",
    "Ladakh":                    "Leh",
}

# Open-Meteo weather codes → human description
WMO_CODES = {
    0: ("Clear Sky", "☀️"),
    1: ("Mainly Clear", "🌤️"),
    2: ("Partly Cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Foggy", "🌫️"),
    48: ("Icy Fog", "🌫️"),
    51: ("Light Drizzle", "🌦️"),
    53: ("Drizzle", "🌦️"),
    55: ("Heavy Drizzle", "🌧️"),
    61: ("Slight Rain", "🌧️"),
    63: ("Moderate Rain", "🌧️"),
    65: ("Heavy Rain", "🌧️"),
    71: ("Slight Snow", "❄️"),
    73: ("Moderate Snow", "❄️"),
    75: ("Heavy Snow", "❄️"),
    80: ("Showers", "🌦️"),
    81: ("Moderate Showers", "🌧️"),
    82: ("Violent Showers", "⛈️"),
    95: ("Thunderstorm", "⛈️"),
    99: ("Hailstorm", "⛈️"),
}


def fetch_weather(state_name: str) -> dict:
    """
    Fetch current weather for the state capital using Open-Meteo (no API key needed).
    Returns a dict with temperature, humidity, precipitation, description, emoji.
    Falls back to dataset average if the request fails.
    """
    city = STATE_CAPITALS.get(state_name, state_name)
    avg_rain = round(state_avg_rainfall.get(state_name, 1000), 1)

    try:
        # 1. Geocoding
        geo_url = (
            f"https://geocoding-api.open-meteo.com/v1/search"
            f"?name={city}&count=1&language=en&format=json"
        )
        geo_resp = requests.get(geo_url, timeout=5)
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            raise ValueError("City not found")

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        # 2. Current weather
        wx_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,precipitation,weather_code"
            f"&timezone=auto"
        )
        wx_resp = requests.get(wx_url, timeout=5)
        wx_data = wx_resp.json()["current"]

        code = int(wx_data.get("weather_code", 0))
        desc, emoji = WMO_CODES.get(code, ("Unknown", "🌡️"))

        return {
            "city":        city,
            "temperature": wx_data.get("temperature_2m", "N/A"),
            "humidity":    wx_data.get("relative_humidity_2m", "N/A"),
            "precipitation": wx_data.get("precipitation", 0),
            "description": desc,
            "emoji":       emoji,
            "avg_rainfall": avg_rain,
            "success":     True,
        }

    except Exception as e:
        return {
            "city":        city,
            "temperature": "N/A",
            "humidity":    "N/A",
            "precipitation": 0,
            "description": "Data unavailable",
            "emoji":       "🌡️",
            "avg_rainfall": avg_rain,
            "success":     False,
            "error":       str(e),
        }


def suggest_crops(state: str, season: str, low_yield_threshold: float, top_n: int = 5) -> list:
    """
    Return top-N crops ranked by average yield for the given state & season.
    Only crops whose mean yield > low_yield_threshold are relevant, but we
    return ALL top-N so the user can see better alternatives.
    """
    season_stripped = season.strip()
    mask = (
        (df_clean["State"] == state) &
        (df_clean["Season"].str.strip() == season_stripped)
    )
    subset = df_clean[mask]

    if subset.empty:
        # Fall back to state only
        subset = df_clean[df_clean["State"] == state]

    if subset.empty:
        return []

    top = (
        subset.groupby("Crop")["Yield"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    top.columns = ["crop", "avg_yield"]
    top["avg_yield"] = top["avg_yield"].round(2)
    return top.to_dict("records")


# -------------------------
# Routes
# -------------------------

@app.route("/")
def home():
    return render_template("index.html", states=states, crops=crops, seasons=seasons)


@app.route("/weather/<state>")
def weather(state):
    """AJAX endpoint — returns JSON weather for the selected state."""
    data = fetch_weather(state)
    return jsonify(data)


@app.route("/predict", methods=["POST"])
def predict():
    state_name  = request.form["state"]
    crop_name   = request.form["crop"]
    season_name = request.form["season"]

    year       = float(request.form["year"])
    area       = float(request.form["area"])
    #production = float(request.form["production"])
    fertilizer = float(request.form["fertilizer"])
    pesticide  = float(request.form["pesticide"])

    # Use rainfall from form (auto-filled by JS from weather endpoint)
    rainfall = float(request.form.get("rainfall", state_avg_rainfall.get(state_name, 1000)))

    # Encode
    state_enc  = le_state.transform([state_name])[0]
    crop_enc   = le_crop.transform([crop_name])[0]
    season_enc = le_season.transform([season_name])[0]

    features = np.array([[
        crop_enc, year, season_enc, state_enc,
        area, rainfall, fertilizer, pesticide
    ]])

    prediction = model.predict(features)[0]

    # Determine if yield is "low" (below 25th percentile of dataset)
    low_threshold = df_clean["Yield"].quantile(0.25)
    is_low = prediction < low_threshold

    suggestions = []
    if is_low:
        suggestions = suggest_crops(state_name, season_name, low_threshold)
        # Remove the selected crop from suggestions if present
        suggestions = [s for s in suggestions if s["crop"].strip() != crop_name.strip()]

    weather_data = fetch_weather(state_name)

    return render_template(
        "index.html",
        states=states,
        crops=crops,
        seasons=seasons,
        prediction=round(prediction, 3),
        is_low=is_low,
        suggestions=suggestions,
        weather=weather_data,
        # re-populate form values
        sel_state=state_name,
        sel_crop=crop_name,
        sel_season=season_name,
        sel_year=int(year),
        sel_area=area,
        sel_rainfall=rainfall,
        sel_fertilizer=fertilizer,
        sel_pesticide=pesticide,
    )


if __name__ == "__main__":
    app.run(debug=True)
