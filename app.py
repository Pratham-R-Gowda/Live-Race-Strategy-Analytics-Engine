import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. PAGE SETUP & NAVIGATION
# ==========================================
st.set_page_config(page_title="F1 Analytics Engine", page_icon="🏎️", layout="centered")

# Sidebar Navigation
st.sidebar.title("🏎️ F1 Pit Wall")
app_mode = st.sidebar.radio("Select Tool:", ["Lap Time Predictor", "Pit Crew Anomaly Detector"])
st.sidebar.divider()

# ==========================================
# 2. LOAD ALL MODELS
# ==========================================
@st.cache_resource
def load_assets():
    # Load XGBoost Lap Time Assets
    xgb_model = joblib.load('xgb_lap_time_model.pkl')
    cols = joblib.load('model_feature_columns.pkl')
    
    # Load Isolation Forest Anomaly Asset
    iso_model = joblib.load('iso_forest_model.pkl')
    
    return xgb_model, cols, iso_model

try:
    xgb_model, feature_columns, iso_forest = load_assets()
except FileNotFoundError:
    st.error("❌ Model files not found! Make sure you exported both the XGBoost and Isolation Forest .pkl files from Jupyter.")
    st.stop()


# ==========================================
# TOOL 1: LAP TIME PREDICTOR (Your original code)
# ==========================================
if app_mode == "Lap Time Predictor":
    st.title("⏱️ Live Race Strategy Engine")
    st.markdown("Adjust the live race conditions below to predict the exact lap time using XGBoost.")
    
    st.sidebar.header("🔧 XGBoost Controls")
    selected_driver = st.sidebar.selectbox("Driver", ["VER", "PER", "ALO", "SAI", "HAM", "STR", "RUS", "BOT", "GAS", "ALB"])
    selected_compound = st.sidebar.selectbox("Tire Compound", ["SOFT", "MEDIUM", "HARD"])

    lap_num = st.sidebar.slider("Lap Number (Fuel Load Proxy)", min_value=1, max_value=57, value=10)
    tyre_life = st.sidebar.slider("Tire Age (Laps)", min_value=1, max_value=40, value=5)
    track_temp = st.sidebar.slider("Track Temperature (°C)", min_value=25.0, max_value=45.0, value=32.0, step=0.5)

    def format_stopwatch(raw_seconds):
        minutes = int(raw_seconds // 60)
        seconds = raw_seconds % 60
        return f"{minutes}:{seconds:06.3f}"

    # Build the exact dataframe the model expects
    input_data = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
    input_data['LapNumber'] = lap_num
    input_data['TyreLife'] = tyre_life
    input_data['TrackTemp'] = track_temp

    compound_col = f'Compound_{selected_compound}'
    if compound_col in input_data.columns:
        input_data[compound_col] = 1

    driver_col = f'Driver_{selected_driver}'
    if driver_col in input_data.columns:
        input_data[driver_col] = 1

    st.divider()
    st.subheader(f"Prediction for {selected_driver}")

    prediction_sec = xgb_model.predict(input_data)[0]
    formatted_time = format_stopwatch(prediction_sec)

    col1, col2 = st.columns(2)
    col1.metric(label="Expected Lap Time", value=formatted_time)
    col2.metric(label="Raw Seconds", value=f"{prediction_sec:.3f}s")
    st.markdown(f"**Conditions:** Lap {lap_num} | {selected_compound} Tires (Age: {tyre_life}) | Temp: {track_temp}°C")


# ==========================================
# TOOL 2: PIT CREW ANOMALY DETECTOR
# ==========================================
elif app_mode == "Pit Crew Anomaly Detector":
    st.title("🚨 Pit Stop Anomaly Detector")
    st.markdown("Enter the total pit lane duration below. The **Isolation Forest** will instantly flag if the stop was normal or botched due to mechanic error.")
    
    st.sidebar.header("🔧 Anomaly Controls")
    pit_driver = st.sidebar.text_input("Driver Initials (e.g., VER, LEC)", value="VER").upper()
    pit_lap = st.sidebar.number_input("Lap Number", min_value=1, max_value=80, value=15)
    
    # Standard pit lane times are usually between 20 and 30 seconds
    pit_duration = st.slider("Total Pit Lane Duration (Seconds)", min_value=15.0, max_value=60.0, value=24.5, step=0.1)
    
    st.divider()
    st.subheader(f"Live Feed: {pit_driver} (Lap {pit_lap})")
    st.write(f"⏱️ **Clocked Time:** {pit_duration:.2f} seconds")
    
    # Run the Isolation Forest inference
    input_matrix = pd.DataFrame({'PitLane_Duration_Sec': [pit_duration]})
    prediction = iso_forest.predict(input_matrix)[0]
    
    if prediction == 1:
        st.success("✅ **STATUS: CLEAN STOP.** Normal variance detected. Send them out!")
        st.balloons() # Just a fun Streamlit visual for a good stop
    else:
        st.error("🚨 **STATUS: ANOMALY DETECTED!** Severe time loss. Review footage for wheel gun or jack failures.")