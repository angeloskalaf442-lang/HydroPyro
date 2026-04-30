
#!/usr/bin/env python3
import os
import sys
import subprocess
import requests
import numpy as np
import pandas as pd
import json
from io import BytesIO
from datetime import datetime, timedelta
from PIL import Image

def ensure_packages():
    required = ["numpy", "pandas", "requests", "pillow", "tensorflow", "folium", "scikit-learn"]
    for pkg in required:
        try:
            __import__(pkg if pkg != "pillow" else "PIL")
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

ensure_packages()

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
import folium

IMG_SIZE = (224, 224)
LSTM_SEQ_LEN = 30
LSTM_FEATURES = 8
MODEL_OUT = "hydropyro_3class_model.keras"
HEADERS = {"User-Agent": "HydroPyro-MVP/2.0"}

# 0 = Normal, 1 = Fire, 2 = Flood
CLASS_NAMES = {
    0: "NORMAL",
    1: "FIRE",
    2: "FLOOD"
}

WEATHER_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "precipitation",
    "windspeed_10m",
    "et0_fao_evapotranspiration",
    "surface_pressure",
    "soil_moisture_0_to_7cm"
]

WEATHER_MIN_MAX = {
    "temperature_2m": (-10, 50),
    "relative_humidity_2m": (0, 100),
    "dewpoint_2m": (-10, 30),
    "precipitation": (0, 50),
    "windspeed_10m": (0, 100),
    "et0_fao_evapotranspiration": (0, 15),
    "surface_pressure": (950, 1050),
    "soil_moisture_0_to_7cm": (0, 1)
}

DISASTER_SAMPLES = [
    # FIRE = 1
    {"name": "Evia Fire 2021", "lat": 38.9, "lon": 23.3, "date": "2021-08-06", "label": 1},
    {"name": "Rhodes Fire 2023", "lat": 36.1, "lon": 27.9, "date": "2023-07-23", "label": 1},
    {"name": "Alexandroupoli Fire 2023", "lat": 40.9, "lon": 25.9, "date": "2023-08-22", "label": 1},
    {"name": "Penteli Fire 2022", "lat": 38.1, "lon": 23.9, "date": "2022-07-19", "label": 1},
    {"name": "Mati Fire 2018", "lat": 38.0, "lon": 23.9, "date": "2018-07-23", "label": 1},
    {"name": "Varibobi Fire 2021", "lat": 38.2, "lon": 23.6, "date": "2021-08-03", "label": 1},
    {"name": "Ilia Fire 2007", "lat": 37.7, "lon": 21.5, "date": "2007-08-24", "label": 1},
    {"name": "Crete Fire 2022", "lat": 35.2, "lon": 25.1, "date": "2022-07-15", "label": 1},
    {"name": "Chios Fire 2012", "lat": 38.4, "lon": 26.1, "date": "2012-08-18", "label": 1},
    {"name": "Zakynthos Fire 2020", "lat": 37.8, "lon": 20.8, "date": "2020-08-10", "label": 1},
    {"name": "Lesvos Fire 2022", "lat": 39.1, "lon": 26.5, "date": "2022-08-12", "label": 1},
    {"name": "Thasos Fire 2016", "lat": 40.7, "lon": 24.7, "date": "2016-09-10", "label": 1},
    {"name": "Peloponnese Fire 2021", "lat": 37.4, "lon": 22.1, "date": "2021-08-10", "label": 1},
    {"name": "Attica Fire 2023", "lat": 38.1, "lon": 23.7, "date": "2023-07-20", "label": 1},
    {"name": "Corfu Fire 2023", "lat": 39.6, "lon": 19.9, "date": "2023-07-25", "label": 1},
    {"name": "Samos Fire 2021", "lat": 37.7, "lon": 26.9, "date": "2021-08-15", "label": 1},
    {"name": "Kavala Fire 2022", "lat": 40.9, "lon": 24.4, "date": "2022-07-18", "label": 1},
    {"name": "Messinia Fire 2020", "lat": 37.1, "lon": 22.0, "date": "2020-08-05", "label": 1},

    # FLOOD = 2
    {"name": "Thessaly Flood 2023", "lat": 39.4, "lon": 22.1, "date": "2023-09-07", "label": 2},
    {"name": "Daniel Flood Volos 2023", "lat": 39.3, "lon": 22.9, "date": "2023-09-05", "label": 2},
    {"name": "Mandra Flood 2017", "lat": 38.1, "lon": 23.5, "date": "2017-11-15", "label": 2},
    {"name": "Karditsa Flood 2020", "lat": 39.4, "lon": 21.9, "date": "2020-09-19", "label": 2},
    {"name": "Crete Flood 2019", "lat": 35.3, "lon": 25.1, "date": "2019-02-17", "label": 2},
    {"name": "Evros Flood 2021", "lat": 41.0, "lon": 26.2, "date": "2021-01-12", "label": 2},
    {"name": "Athens Flood 2014", "lat": 37.98, "lon": 23.72, "date": "2014-10-24", "label": 2},
    {"name": "Patras Flood 2021", "lat": 38.25, "lon": 21.73, "date": "2021-11-05", "label": 2},
    {"name": "Larissa Flood 2018", "lat": 39.64, "lon": 22.42, "date": "2018-03-03", "label": 2},
    {"name": "Rhodes Flood 2013", "lat": 36.4, "lon": 28.2, "date": "2013-11-22", "label": 2},
    {"name": "Corfu Flood 2021", "lat": 39.6, "lon": 19.9, "date": "2021-10-15", "label": 2},
    {"name": "Ioannina Flood 2015", "lat": 39.66, "lon": 20.85, "date": "2015-02-01", "label": 2},
    {"name": "Kalamata Flood 2016", "lat": 37.04, "lon": 22.11, "date": "2016-09-07", "label": 2},
    {"name": "Chania Flood 2019", "lat": 35.51, "lon": 24.02, "date": "2019-02-25", "label": 2},
    {"name": "Drama Flood 2022", "lat": 41.15, "lon": 24.15, "date": "2022-12-10", "label": 2},
    {"name": "Serres Flood 2020", "lat": 41.09, "lon": 23.55, "date": "2020-01-05", "label": 2},
    {"name": "Xanthi Flood 2017", "lat": 41.14, "lon": 24.88, "date": "2017-02-02", "label": 2},
    {"name": "Heraklion Flood 2022", "lat": 35.34, "lon": 25.14, "date": "2022-10-15", "label": 2},
]

NORMAL_SAMPLES = [
    {"name": "Normal Athens Spring", "lat": 37.98, "lon": 23.72, "date": "2022-04-10", "label": 0},
    {"name": "Normal Thessaloniki Spring", "lat": 40.64, "lon": 22.94, "date": "2022-05-12", "label": 0},
    {"name": "Normal Larissa Spring", "lat": 39.64, "lon": 22.42, "date": "2021-04-20", "label": 0},
    {"name": "Normal Volos Spring", "lat": 39.36, "lon": 22.94, "date": "2021-05-05", "label": 0},
    {"name": "Normal Patras Spring", "lat": 38.25, "lon": 21.73, "date": "2020-04-18", "label": 0},
    {"name": "Normal Heraklion Spring", "lat": 35.34, "lon": 25.14, "date": "2021-04-14", "label": 0},
    {"name": "Normal Rhodes Spring", "lat": 36.43, "lon": 28.22, "date": "2021-05-03", "label": 0},
    {"name": "Normal Corfu Spring", "lat": 39.62, "lon": 19.92, "date": "2022-05-09", "label": 0},
    {"name": "Normal Ioannina Spring", "lat": 39.66, "lon": 20.85, "date": "2022-04-22", "label": 0},
    {"name": "Normal Kavala Spring", "lat": 40.94, "lon": 24.41, "date": "2021-05-15", "label": 0},
    {"name": "Normal Kalamata Spring", "lat": 37.04, "lon": 22.11, "date": "2020-05-08", "label": 0},
    {"name": "Normal Chania Spring", "lat": 35.51, "lon": 24.02, "date": "2022-04-17", "label": 0},
    {"name": "Normal Karditsa Spring", "lat": 39.36, "lon": 21.92, "date": "2021-04-11", "label": 0},
    {"name": "Normal Serres Spring", "lat": 41.09, "lon": 23.55, "date": "2022-05-17", "label": 0},
    {"name": "Normal Drama Spring", "lat": 41.15, "lon": 24.15, "date": "2021-05-20", "label": 0},
    {"name": "Normal Xanthi Spring", "lat": 41.14, "lon": 24.88, "date": "2020-04-25", "label": 0},
    {"name": "Normal Samos Spring", "lat": 37.75, "lon": 26.97, "date": "2022-05-02", "label": 0},
    {"name": "Normal Zakynthos Spring", "lat": 37.79, "lon": 20.89, "date": "2021-04-29", "label": 0},
]

TRAINING_SAMPLES = DISASTER_SAMPLES + NORMAL_SAMPLES

def get_coords(query: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query, "count": 1, "language": "el", "format": "json"}
    r = requests.get(url, params=params, headers=HEADERS, timeout=10)
    r.raise_for_status()
    data = r.json().get("results")
    if not data:
        raise Exception("Η περιοχή δεν βρέθηκε.")
    return float(data[0]["latitude"]), float(data[0]["longitude"]), data[0]["name"]

def fetch_weather_dataframe(lat, lon, date_obj):
    now = datetime.now()
    is_future = date_obj.date() > now.date()

    url = "https://api.open-meteo.com/v1/forecast" if is_future else "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_COLS),
        "timezone": "UTC"
    }

    if is_future:
        params["forecast_days"] = 3
    else:
        params["start_date"] = (date_obj - timedelta(days=2)).strftime("%Y-%m-%d")
        params["end_date"] = date_obj.strftime("%Y-%m-%d")

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        hourly = r.json().get("hourly", {})
        if not hourly:
            return pd.DataFrame(columns=WEATHER_COLS)
        df = pd.DataFrame(hourly)
        return df[WEATHER_COLS].ffill().bfill()
    except Exception:
        return pd.DataFrame(columns=WEATHER_COLS)

def normalize_weather(df):
    if df.empty:
        return np.zeros((LSTM_SEQ_LEN, LSTM_FEATURES), dtype=np.float32)

    norm = df.copy()
    for col in WEATHER_COLS:
        mi, ma = WEATHER_MIN_MAX[col]
        norm[col] = (norm[col] - mi) / (ma - mi + 1e-7)

    data = np.nan_to_num(norm.values.astype(np.float32))

    if len(data) < LSTM_SEQ_LEN:
        pad = np.zeros((LSTM_SEQ_LEN - len(data), LSTM_FEATURES), dtype=np.float32)
        data = np.vstack([pad, data])

    return data[-LSTM_SEQ_LEN:]

def fetch_nasa_image(lat, lon, date_obj):
    q_date = min(date_obj, datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    url = "https://wvs.earthdata.nasa.gov/api/v1/snapshot"
    box = 0.2

    params = {
        "REQUEST": "GetSnapshot",
        "TIME": q_date,
        "BBOX": f"{lon-box},{lat-box},{lon+box},{lat+box}",
        "CRS": "EPSG:4326",
        "LAYERS": "MODIS_Terra_CorrectedReflectance_TrueColor",
        "FORMAT": "image/jpeg",
        "WIDTH": 224,
        "HEIGHT": 224
    }

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return np.array(img.resize(IMG_SIZE), dtype=np.float32) / 255.0
    except Exception:
        return np.zeros((224, 224, 3), dtype=np.float32)

def augment_image(img):
    aug = img.copy()

    if np.random.rand() > 0.5:
        aug = np.fliplr(aug)

    brightness = np.random.uniform(0.9, 1.1)
    aug = np.clip(aug * brightness, 0, 1)

    noise = np.random.normal(0, 0.015, aug.shape)
    aug = np.clip(aug + noise, 0, 1)

    return aug.astype(np.float32)

def augment_weather(weather):
    aug = weather.copy()
    noise = np.random.normal(0, 0.02, aug.shape)
    aug = np.clip(aug + noise, 0, 1)
    return aug.astype(np.float32)

def build_hybrid_model():
    img_in = layers.Input(shape=(224, 224, 3), name="image_input")

    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal")(img_in)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(96, (3, 3), activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    seq_in = layers.Input(shape=(LSTM_SEQ_LEN, LSTM_FEATURES), name="sensor_input")

    y = layers.LSTM(64, return_sequences=True)(seq_in)
    y = layers.LSTM(32)(y)

    combined = layers.Concatenate()([x, y])

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.35)(z)
    z = layers.Dense(48, activation="relu")(z)
    z = layers.Dropout(0.25)(z)

    out = layers.Dense(3, activation="softmax", name="output")(z)

    model = models.Model(inputs=[img_in, seq_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def train_model():
    print("\n🚀 Εκκίνηση εκπαίδευσης HydroPyro 3-class model...")
    print("Classes: 0=NORMAL, 1=FIRE, 2=FLOOD")

    model = build_hybrid_model()

    base_img, base_weather, base_labels = [], [], []

    for s in TRAINING_SAMPLES:
        print(f" -> Fetching: {s['name']}")
        d = datetime.strptime(s["date"], "%Y-%m-%d")
        img = fetch_nasa_image(s["lat"], s["lon"], d)
        raw_weather = fetch_weather_dataframe(s["lat"], s["lon"], d)
        weather = normalize_weather(raw_weather)

        base_img.append(img)
        base_weather.append(weather)
        base_labels.append(s["label"])

    x_img, x_weather, y_labels = [], [], []

    # Data augmentation: κάθε sample γίνεται περισσότερα training examples
    AUG_PER_SAMPLE = 5

    for img, weather, label in zip(base_img, base_weather, base_labels):
        x_img.append(img)
        x_weather.append(weather)
        y_labels.append(label)

        for _ in range(AUG_PER_SAMPLE):
            x_img.append(augment_image(img))
            x_weather.append(augment_weather(weather))
            y_labels.append(label)

    x_img = np.array(x_img, dtype=np.float32)
    x_weather = np.array(x_weather, dtype=np.float32)
    y_labels = np.array(y_labels, dtype=np.int32)

    classes = np.unique(y_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_labels
    )

    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    print("\n📊 Training distribution:")
    for cls in classes:
        print(f" {CLASS_NAMES[int(cls)]}: {(y_labels == cls).sum()} samples")

    print("\n⚖️ Class weights:")
    print(class_weight)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=8,
            restore_best_weights=True
        )
    ]

    model.fit(
        {
            "image_input": x_img,
            "sensor_input": x_weather
        },
        y_labels,
        epochs=50,
        batch_size=8,
        verbose=1,
        class_weight=class_weight,
        callbacks=callbacks
    )

    model.save(MODEL_OUT)
    print(f"\n✅ Model saved as: {MODEL_OUT}")
    return model

def classify_level(prob, predicted_class):
    if predicted_class == 0:
        if prob >= 0.75:
            return "LOW"
        return "UNCERTAIN"

    if prob >= 0.85:
        return "EXTREME"
    elif prob >= 0.70:
        return "HIGH"
    elif prob >= 0.45:
        return "MEDIUM"
    return "LOW"

def get_color(level):
    return {
        "LOW": "green",
        "UNCERTAIN": "blue",
        "MEDIUM": "orange",
        "HIGH": "red",
        "EXTREME": "darkred"
    }.get(level, "blue")

def confidence_score(probs):
    sorted_probs = sorted(probs, reverse=True)
    gap = sorted_probs[0] - sorted_probs[1]

    if gap >= 0.45:
        return "HIGH"
    elif gap >= 0.20:
        return "MEDIUM"
    return "LOW"

def detect_weather_trend(df, dominant_risk):
    if df.empty or len(df) < 8:
        return "INSUFFICIENT_DATA"

    recent = df.tail(8)
    previous = df.iloc[:-8]

    if previous.empty:
        return "INSUFFICIENT_DATA"

    if dominant_risk == "FLOOD":
        recent_rain = recent["precipitation"].mean()
        previous_rain = previous["precipitation"].mean()

        if recent_rain > previous_rain * 1.5 and recent_rain > 1:
            return "INCREASING"
        elif recent_rain < previous_rain * 0.7:
            return "DECREASING"
        return "STABLE"

    if dominant_risk == "FIRE":
        recent_temp = recent["temperature_2m"].mean()
        previous_temp = previous["temperature_2m"].mean()
        recent_rh = recent["relative_humidity_2m"].mean()
        previous_rh = previous["relative_humidity_2m"].mean()

        if recent_temp > previous_temp + 2 and recent_rh < previous_rh:
            return "INCREASING"
        elif recent_temp < previous_temp - 2 or recent_rh > previous_rh:
            return "DECREASING"
        return "STABLE"

    return "STABLE"

def detect_anomaly(df, dominant_risk):
    if df.empty:
        return "UNKNOWN"

    rain = df["precipitation"].sum()
    temp = df["temperature_2m"].max()
    rh = df["relative_humidity_2m"].min()
    wind = df["windspeed_10m"].max()
    soil = df["soil_moisture_0_to_7cm"].mean()

    if dominant_risk == "FLOOD":
        if rain >= 80 or soil >= 0.75:
            return "EXTREME_RAINFALL_OR_SOIL_MOISTURE_ANOMALY"
        elif rain >= 30:
            return "HEAVY_RAINFALL_ANOMALY"
        return "NO_MAJOR_ANOMALY"

    if dominant_risk == "FIRE":
        if temp >= 38 and rh <= 25 and wind >= 35:
            return "EXTREME_FIRE_WEATHER_ANOMALY"
        elif temp >= 32 and rh <= 35:
            return "FIRE_WEATHER_ANOMALY"
        return "NO_MAJOR_ANOMALY"

    return "NO_MAJOR_ANOMALY"

def generate_recommendation(dominant_risk, level):
    if dominant_risk == "NORMAL":
        return "No immediate disaster-prevention action required. Continue routine environmental monitoring."

    if dominant_risk == "FLOOD":
        if level in ["HIGH", "EXTREME"]:
            return (
                "Increase monitoring of drainage systems, low-lying zones, riverbeds and vulnerable infrastructure. "
                "Prepare preventive civil-protection actions."
            )
        elif level == "MEDIUM":
            return "Monitor rainfall evolution and inspect flood-prone locations."
        return "Continue routine monitoring."

    if dominant_risk == "FIRE":
        if level in ["HIGH", "EXTREME"]:
            return (
                "Increase surveillance of vegetation zones, monitor wind evolution, prepare firefighting resources "
                "and consider preventive restrictions on risky outdoor activity."
            )
        elif level == "MEDIUM":
            return "Monitor temperature, humidity and wind conditions. Prepare preventive checks."
        return "Continue routine monitoring."

    return "Continue monitoring."

def generate_justification(df, dominant_risk):
    if df.empty:
        return "Risk output generated with limited weather data availability."

    temp_max = df["temperature_2m"].max()
    rh_min = df["relative_humidity_2m"].min()
    rain_sum = df["precipitation"].sum()
    wind_max = df["windspeed_10m"].max()
    soil_mean = df["soil_moisture_0_to_7cm"].mean()

    if dominant_risk == "NORMAL":
        return (
            f"Normal-risk output is supported by non-extreme observed conditions: "
            f"{rain_sum:.1f} mm accumulated precipitation, {temp_max:.1f}°C maximum temperature, "
            f"{rh_min:.1f}% minimum relative humidity, {wind_max:.1f} km/h maximum windspeed "
            f"and {soil_mean:.2f} mean near-surface soil moisture."
        )

    if dominant_risk == "FLOOD":
        return (
            f"Flood risk is supported by accumulated precipitation of {rain_sum:.1f} mm, "
            f"maximum windspeed of {wind_max:.1f} km/h and mean near-surface soil moisture of {soil_mean:.2f}."
        )

    if dominant_risk == "FIRE":
        return (
            f"Fire risk is supported by maximum temperature of {temp_max:.1f}°C, "
            f"minimum relative humidity of {rh_min:.1f}%, maximum windspeed of {wind_max:.1f} km/h "
            f"and accumulated precipitation of {rain_sum:.1f} mm."
        )

    return "Risk justification unavailable."

def generate_alert(dominant_risk, level, city):
    if dominant_risk == "NORMAL":
        return f"✅ NORMAL CONDITIONS in {city}. Routine monitoring is sufficient."

    if level == "EXTREME":
        return f"🚨 EXTREME {dominant_risk} RISK in {city}. Immediate preparedness review recommended."
    if level == "HIGH":
        return f"⚠️ HIGH {dominant_risk} RISK in {city}. Preventive monitoring recommended."
    if level == "MEDIUM":
        return f"⚠️ MEDIUM {dominant_risk} RISK in {city}. Conditions should be monitored."

    return f"✅ LOW {dominant_risk} RISK in {city}. Routine monitoring is sufficient."

def priority_rank(level, dominant_risk):
    if dominant_risk == "NORMAL":
        return 5

    return {
        "EXTREME": 1,
        "HIGH": 2,
        "MEDIUM": 3,
        "LOW": 4,
        "UNCERTAIN": 4
    }.get(level, 4)

def create_map(lat, lon, city, output):
    level = output["risk_level"]
    color = get_color(level)

    m = folium.Map(location=[lat, lon], zoom_start=9)

    popup = f"""
    <b>{city}</b><br>
    Dominant output: {output["dominant_output"]}<br>
    Normal probability: {output["risk_scores"]["normal_probability"] * 100:.1f}%<br>
    Fire probability: {output["risk_scores"]["fire_probability"] * 100:.1f}%<br>
    Flood probability: {output["risk_scores"]["flood_probability"] * 100:.1f}%<br>
    Risk level: {output["risk_level"]}<br>
    Trend: {output["trend"]}<br>
    Priority rank: {output["priority_rank"]}
    """

    folium.CircleMarker(
        location=[lat, lon],
        radius=20,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.65,
        popup=popup
    ).add_to(m)

    filename = f"hydropyro_map_{city.lower().replace(' ', '_')}.html"
    m.save(filename)
    return filename

def build_hydropyro_output(city, lat, lon, date_str, probs, raw_weather_df):
    normal_prob = float(probs[0])
    fire_prob = float(probs[1])
    flood_prob = float(probs[2])

    predicted_class = int(np.argmax(probs))
    dominant_output = CLASS_NAMES[predicted_class]
    dominant_probability = float(np.max(probs))

    level = classify_level(dominant_probability, predicted_class)
    trend = detect_weather_trend(raw_weather_df, dominant_output)
    anomaly = detect_anomaly(raw_weather_df, dominant_output)
    confidence = confidence_score(probs)

    output = {
        "location": {
            "name": city,
            "latitude": lat,
            "longitude": lon
        },
        "date": date_str,
        "risk_scores": {
            "normal_probability": round(normal_prob, 4),
            "fire_probability": round(fire_prob, 4),
            "flood_probability": round(flood_prob, 4)
        },
        "dominant_output": dominant_output,
        "dominant_probability": round(dominant_probability, 4),
        "risk_level": level,
        "priority_rank": priority_rank(level, dominant_output),
        "alert": generate_alert(dominant_output, level, city),
        "trend": trend,
        "anomaly": anomaly,
        "confidence": confidence,
        "recommended_action": generate_recommendation(dominant_output, level),
        "decision_justification": generate_justification(raw_weather_df, dominant_output),
        "model_note": (
            "HydroPyro MVP output. This is a decision-support estimate and must not replace "
            "official meteorological, hydrological or civil-protection assessment."
        )
    }

    return output

def print_report(output, map_file, json_file):
    city = output["location"]["name"]

    print(f"\n📊 HYDROPYRO RISK INTELLIGENCE REPORT — {city.upper()}")
    print("-" * 65)
    print(f"📅 Date: {output['date']}")
    print(f"✅ Normal probability: {output['risk_scores']['normal_probability']:.1%}")
    print(f"🔥 Fire probability: {output['risk_scores']['fire_probability']:.1%}")
    print(f"🌊 Flood probability: {output['risk_scores']['flood_probability']:.1%}")
    print(f"🎯 Dominant output: {output['dominant_output']}")
    print(f"🚦 Risk level: {output['risk_level']}")
    print(f"📌 Priority rank: {output['priority_rank']}")
    print(f"📈 Trend: {output['trend']}")
    print(f"🧪 Anomaly: {output['anomaly']}")
    print(f"✅ Confidence: {output['confidence']}")
    print(f"\n🚨 Alert:\n{output['alert']}")
    print(f"\n🛠 Recommended action:\n{output['recommended_action']}")
    print(f"\n🧾 Decision justification:\n{output['decision_justification']}")
    print(f"\n🗺 Map saved as: {map_file}")
    print(f"📁 JSON saved as: {json_file}")

def save_json(output, city):
    safe_city = city.lower().replace(" ", "_")
    filename = f"hydropyro_output_{safe_city}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return filename

def main():
    print("\n--- 🛡️ HYDROPYRO MVP 3-CLASS RISK INTELLIGENCE ---")
    print("Outputs: NORMAL / FIRE / FLOOD")

    retrain = input("\n🔁 Retrain model? yes/no: ").strip().lower()

    if retrain == "yes" or not os.path.exists(MODEL_OUT):
        model = train_model()
    else:
        model = tf.keras.models.load_model(MODEL_OUT)

    place = input("\n📍 Περιοχή: ").strip()

    try:
        lat, lon, name = get_coords(place)

        date_str = input("📅 Ημερομηνία Πρόβλεψης (YYYY-MM-DD) ή Enter: ").strip()
        t_date = datetime.strptime(date_str, "%Y-%m-%d") if date_str else datetime.now()
        date_str = t_date.strftime("%Y-%m-%d")

        print("\n📡 Fetching environmental data...")
        img = fetch_nasa_image(lat, lon, t_date)
        raw_weather_df = fetch_weather_dataframe(lat, lon, t_date)
        weather = normalize_weather(raw_weather_df)

        print("🧠 Running HydroPyro 3-class model...")
        probs = model.predict(
            {
                "image_input": img[None, ...],
                "sensor_input": weather[None, ...]
            },
            verbose=0
        )[0]

        output = build_hydropyro_output(
            city=name,
            lat=lat,
            lon=lon,
            date_str=date_str,
            probs=probs,
            raw_weather_df=raw_weather_df
        )

        map_file = create_map(lat, lon, name, output)
        json_file = save_json(output, name)

        print_report(output, map_file, json_file)

    except Exception as e:
        print(f"❌ Σφάλμα: {e}")

if __name__ == "__main__":
    main()
