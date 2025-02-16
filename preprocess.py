import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_data(df, train=False, encoder=None, scaler=None):
    categorical_cols = ["Asthma Symptoms Frequency", "Triggers", "Weather Sensitivity", 
                        "Poor Air Quality Exposure", "Night Breathing Difficulty"]
    numerical_cols = ["AQI", "PM2.5", "SO2 level", "NO2 level", "CO2 level", "Humidity", "Temperature"]

    # Extract Features & Target Variable
    X = df[numerical_cols + categorical_cols]
    y = df["Risk Factor"] if "Risk Factor" in df else None

    # Handle Encoder & Scaler
    if train:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  
        scaler = StandardScaler()
        encoded_cats = encoder.fit_transform(X[categorical_cols])
        scaled_nums = scaler.fit_transform(X[numerical_cols])
    else:
        if encoder is None or scaler is None:
            raise ValueError("Encoder and Scaler must be provided in inference mode")
        encoded_cats = encoder.transform(X[categorical_cols])
        scaled_nums = scaler.transform(X[numerical_cols])

    # Convert Encoded & Scaled Data into DataFrames
    X_processed = pd.DataFrame(scaled_nums, columns=numerical_cols)
    X_encoded = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

    # Concatenate Processed Numerical & Categorical Features
    X_final = pd.concat([X_processed, X_encoded], axis=1)

    return X_final, y, encoder, scaler
