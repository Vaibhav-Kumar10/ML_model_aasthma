import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# File paths to store encoders & scalers
ENCODER_PATH = "encoder.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.pkl"

def preprocess_data(df, train=False):
    """
    Preprocess input data for model inference.
    
    If `train=True`, it fits and saves the encoders/scalers.
    Otherwise, it loads pre-trained ones and ensures consistency.
    """
    categorical_cols = ["Asthma Symptoms Frequency", "Triggers", "Weather Sensitivity", 
                        "Poor Air Quality Exposure", "Night Breathing Difficulty"]
    numerical_cols = ["AQI", "PM2.5", "SO2 level", "NO2 level", "CO2 level", "Humidity", "Temperature"]

    # --- Feature Engineering ---
    df["AQI_PM_Ratio"] = df["AQI"] / (df["PM2.5"] + 1)  # Avoid division by zero
    df["CO2_SO2_Interaction"] = df["CO2 level"] * df["SO2 level"]

    # Expected numerical features
    numerical_cols.extend(["AQI_PM_Ratio", "CO2_SO2_Interaction"])

    # Check if training or inference
    if train:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        scaler = StandardScaler()

        # Fit encoders/scalers
        encoded_cats = encoder.fit_transform(df[categorical_cols])
        scaled_nums = scaler.fit_transform(df[numerical_cols])

        # Save fitted encoders/scalers
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(encoder, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        with open(FEATURES_PATH, "wb") as f:
            pickle.dump(numerical_cols + list(encoder.get_feature_names_out(categorical_cols)), f)
    
    else:  # Inference mode
        # Load pre-trained encoders/scalers
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(FEATURES_PATH, "rb") as f:
            expected_features = pickle.load(f)

        # Ensure all expected features exist
        for col in numerical_cols:
            if col not in df:
                df[col] = 0  # Default missing numerical features to 0

        # Encode and scale
        encoded_cats = encoder.transform(df[categorical_cols])
        scaled_nums = scaler.transform(df[numerical_cols])

    # Convert to DataFrame
    X_processed = pd.DataFrame(scaled_nums, columns=numerical_cols)
    X_encoded = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
    X_final = pd.concat([X_processed, X_encoded], axis=1)

    # Ensure columns match training
    for col in expected_features:
        if col not in X_final:
            X_final[col] = 0  # Fill missing features
    X_final = X_final[expected_features]  # Ensure correct order

    y = df["Risk Factor"] if "Risk Factor" in df else None
    return X_final, y
