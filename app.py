import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load(r"L:\Entertainment\Data science\CCTV Maintanence\Model Building\Models\logistic_pipeline.pkl")

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="CCTV Maintenance Prediction",
    page_icon="📹",
    layout="wide"
)

st.title("📹 CCTV Predictive Maintenance")

st.write(
    "Predict whether a CCTV camera requires maintenance using Machine Learning."
)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Enter Camera Details")

age_days = st.sidebar.number_input(
    "Camera Age (Days)",
    0,
    5000,
    365
)

daily_usage_hours = st.sidebar.number_input(
    "Daily Usage Hours",
    0.0,
    24.0,
    16.0
)

avg_temperature_c = st.sidebar.number_input(
    "Average Temperature (°C)",
    -10.0,
    60.0,
    25.0
)

avg_humidity_percent = st.sidebar.slider(
    "Humidity (%)",
    0,
    100,
    60
)

vibration_level = st.sidebar.number_input(
    "Vibration Level",
    0.0,
    10.0,
    0.5
)

signal_strength_dbm = st.sidebar.number_input(
    "Signal Strength (dBm)",
    -100,
    0,
    -55
)

avg_power_consumption_w = st.sidebar.number_input(
    "Power Consumption (W)",
    0.0,
    100.0,
    15.0
)

firmware_version = st.sidebar.selectbox(
    "Firmware Version",
    list(range(1,11))
)

camera_type = st.sidebar.selectbox(
    "Camera Type",
    [
        "Dome",
        "Bullet",
        "PTZ"
    ]
)

location_type = st.sidebar.selectbox(
    "Location",
    [
        "Indoor",
        "Outdoor"
    ]
)

manufacturer = st.sidebar.selectbox(
    "Manufacturer",
    [
        "Hikvision",
        "Dahua",
        "Axis",
        "Bosch"
    ]
)

weather_exposure = st.sidebar.selectbox(
    "Weather Exposure",
    [
        "Low",
        "Medium",
        "High"
    ]
)

# -----------------------------
# Create Data
# -----------------------------

data = pd.DataFrame({

    "age_days":[age_days],

    "daily_usage_hours":[daily_usage_hours],

    "avg_temperature_c":[avg_temperature_c],

    "avg_humidity_percent":[avg_humidity_percent],

    "vibration_level":[vibration_level],

    "signal_strength_dbm":[signal_strength_dbm],

    "avg_power_consumption_w":[avg_power_consumption_w],

    "firmware_version":[firmware_version],

    "camera_type":[camera_type],

    "location_type":[location_type],

    "manufacturer":[manufacturer],

    "weather_exposure":[weather_exposure]

})

# -----------------------------
# Feature Engineering
# -----------------------------

data["camera_age_years"] = data["age_days"] / 365

data["old_camera"] = (data["age_days"] > 1000).astype(int)

data["high_temp"] = (data["avg_temperature_c"] > 35).astype(int)

data["high_humidity"] = (
    data["avg_humidity_percent"] > 80
).astype(int)

data["weak_signal"] = (
    data["signal_strength_dbm"] < -65
).astype(int)

data["high_usage"] = (
    data["daily_usage_hours"] > 18
).astype(int)

data["power_per_hour"] = (
    data["avg_power_consumption_w"] /
    data["daily_usage_hours"]
)

data["temp_humidity"] = (
    data["avg_temperature_c"] *
    data["avg_humidity_percent"]
)

data["age_vibration"] = (
    data["camera_age_years"] *
    data["vibration_level"]
)

data["signal_quality"] = (
    100 +
    data["signal_strength_dbm"]
)

data["risk_score"] = (

    data["old_camera"]

    + data["high_temp"]

    + data["high_humidity"]

    + data["weak_signal"]

    + data["high_usage"]

)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    if prediction == 1:

        st.error("⚠ Maintenance Required")

    else:

        st.success("✅ Camera is Healthy")

    st.metric(
        "Maintenance Probability",
        f"{probability*100:.2f}%"
    )

    st.subheader("Input Data")

    st.dataframe(data)

