import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_data(df, train=False, encoder=None, scaler=None):
    categorical_cols = ["Asthma Symptoms Frequency", "Triggers", "Weather Sensitivity", 
                        "Poor Air Quality Exposure", "Night Breathing Difficulty"]
    numerical_cols = ["AQI", "PM2.5", "SO2 level", "NO2 level", "CO2 level", "Humidity", "Temperature"]

    # Feature Engineering: Creating New Features
    df["AQI_PM_Ratio"] = df["AQI"] / (df["PM2.5"] + 1)  # Avoid division by zero
    df["CO2_SO2_Interaction"] = df["CO2 level"] * df["SO2 level"]
    numerical_cols.extend(["AQI_PM_Ratio", "CO2_SO2_Interaction"])  # Add new features

    X = df[numerical_cols + categorical_cols]
    y = df["Risk Factor"] if "Risk Factor" in df else None

    if train:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        scaler = StandardScaler()
        encoded_cats = encoder.fit_transform(X[categorical_cols])
        scaled_nums = scaler.fit_transform(X[numerical_cols])
    else:
        encoded_cats = encoder.transform(X[categorical_cols])
        scaled_nums = scaler.transform(X[numerical_cols])

    X_processed = pd.DataFrame(scaled_nums, columns=numerical_cols)
    X_encoded = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
    X_final = pd.concat([X_processed, X_encoded], axis=1)

    return X_final, y, encoder, scaler
