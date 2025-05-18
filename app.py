import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import datetime

# --- Sidebar Configuration ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload", "Model Training", "Evaluation", "Route Suggestion"])

# --- Page 1: Data Upload and Collection ---
if page == "Data Upload":
    st.title("1. Data Collection and Upload")
    st.markdown("""
        Upload your historical logistics data or collect live data from APIs (fleet, weather, maps).
    """)

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Sample Data:", data.head())

        if 'fuel_type' in data.columns:
            fuel_encoding = {"diesel": 0, "petrol": 1, "electric": 2}
            data['fuel_type_encoded'] = data['fuel_type'].map(fuel_encoding)

        if st.button("Normalize & Handle Missing Values"):
            data.fillna(method='ffill', inplace=True)
            features = data[[
                'route_distance_km', 'fuel_type_encoded', 'avg_speed_kmph',
                'cargo_weight_kg', 'temperature_c', 'wind_speed_kmph', 'traffic_index',
                'humidity_percent', 'road_condition_index', 'vehicle_age_years'
            ]]
            target = data['carbon_emission']
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            st.session_state['features'] = scaled_features
            st.session_state['target'] = target.values
            st.session_state['scaler'] = scaler
            st.success("Data Preprocessed Successfully")

# --- Page 2: Model Training ---
elif page == "Model Training":
    st.title("2. Train Carbon Emission Prediction Model")

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.callbacks import EarlyStopping
    import numpy as np

    if 'features' in st.session_state:
        features = st.session_state['features']
        target = st.session_state['target']
        scaler = st.session_state['scaler']

        if st.button("Train Model"):
            # Convert features and target to numpy arrays
            X = np.array(features)
            y = np.array(target)

            # Build a simple feed-forward neural network
            model = Sequential([
                Dense(64, activation='relu', input_shape=(X.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1)  # Output layer for regression
            ])

            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # Use early stopping to prevent overfitting
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            # Train the model (use 20% validation split)
            history = model.fit(
                X, y,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )

            # Save the trained model
            model.save("carbon_model_dl.h5")

            # Save scaler as before
            joblib.dump(scaler, "scaler.pkl")

            st.session_state['model_trained'] = True
            st.success("Deep Learning Model Trained and Saved")

    else:
        st.warning("Please preprocess the data first in the Data Upload section.")

# --- Page 3: Evaluation ---
elif page == "Evaluation":
    st.title("3. Model Evaluation")

    if 'features' in st.session_state and 'model_trained' in st.session_state:
        features = st.session_state['features']
        target = st.session_state['target']
        model = joblib.load("carbon_model.pkl")

        predictions = model.predict(features)
        mae = mean_absolute_error(target, predictions)
        rmse = mean_squared_error(target, predictions, squared=False)
        percent_error = np.mean(np.abs((target - predictions) / target)) * 100

        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
        st.metric("% Error", f"{percent_error:.2f}%")

        st.line_chart(pd.DataFrame({"Actual": target, "Predicted": predictions}))
    else:
        st.warning("Train the model first before evaluation.")

# --- Page 4: Route Suggestion ---
elif page == "Route Suggestion":
    st.title("4. Route Suggestion Engine")
    st.markdown("""
        Enter real-time data to get the most eco-friendly route.
    """)

    if 'model_trained' in st.session_state:
        model = joblib.load("carbon_model.pkl")
        scaler = joblib.load("scaler.pkl")

        # Input features
        route_distance = st.number_input("Route Distance (km)", min_value=1.0)
        fuel_type = st.selectbox("Fuel Type", ["diesel", "petrol", "electric"])
        avg_speed = st.number_input("Average Speed (km/h)", min_value=1.0)
        cargo_weight = st.number_input("Cargo Weight (kg)", min_value=0.0)
        temperature = st.number_input("Temperature (°C)")
        wind_speed = st.number_input("Wind Speed (km/h)")
        traffic_index = st.slider("Traffic Level (0 - 1)", 0.0, 1.0)
        humidity_percent = st.slider("Humidity (%)", 0, 100)
        road_condition_index = st.slider("Road Condition Index (0: worst - 1: best)", 0.0, 1.0)
        vehicle_age_years = st.number_input("Vehicle Age (years)", min_value=0.0)

        fuel_encoding = {"diesel": 0, "petrol": 1, "electric": 2}
        input_features = np.array([[
            route_distance,
            fuel_encoding[fuel_type],
            avg_speed,
            cargo_weight,
            temperature,
            wind_speed,
            traffic_index,
            humidity_percent,
            road_condition_index,
            vehicle_age_years
        ]])

        input_scaled = scaler.transform(input_features)
        predicted_emission = model.predict(input_scaled)[0]

        st.success(f"Predicted Carbon Emission: {predicted_emission:.2f} gCO₂")
    else:
        st.warning("Train the model first before using the suggestion engine.")
