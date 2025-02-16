import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Load trained model
model = tf.keras.models.load_model("model.keras")

# Load preprocessing objects (encoder & scaler)
with open("preprocessing.pkl", "rb") as f:
    encoder, scaler = pickle.load(f)

# Define Flask app
app = Flask(_name_)

# Define input features expected by the model
FEATURES = [
    "AQI", "PM2.5", "SO2 level", "NO2 level", "CO2 level", 
    "Humidity", "Temperature", "Asthma Symptoms Frequency", 
    "Triggers", "Weather Sensitivity", "Poor Air Quality Exposure", 
    "Night Breathing Difficulty"
]

@app.route("/")
def home():
    return jsonify({"message": "Asthma Risk Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        
        # Ensure all required fields are present
        if not all(feature in data for feature in FEATURES):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Convert JSON data to DataFrame
        df = pd.DataFrame([data])

        # ✅ Add Missing Feature Used in Training
        df["AQI_PM_Ratio"] = df["AQI"] / (df["PM2.5"] + 1)  # Prevents division by zero

        # ✅ Remove Extra Features Not Used in Training
        unwanted_features = ["CO2_Squared", "Log_CO2", "Log_SO2", "SO2_Squared"]
        df = df.drop(columns=[col for col in unwanted_features if col in df.columns], errors='ignore')

        # Preprocess categorical & numerical data
        categorical_features = ["Asthma Symptoms Frequency", "Triggers", "Weather Sensitivity", "Poor Air Quality Exposure", "Night Breathing Difficulty"]
        numerical_features = [col for col in df.columns if col not in categorical_features]

        # Encode categorical features
        df_categorical = encoder.transform(df[categorical_features])  
        df_numerical = scaler.transform(df[numerical_features])

        # Combine transformed features
        X = np.hstack([df_numerical, df_categorical])

        # Make prediction
        prediction = model.predict(X)[0][0]

        # Return prediction
        return jsonify({"asthma_risk_score": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if _name_ == "_main_":
    app.run(host="0.0.0.0", port=7860, debug=True)
