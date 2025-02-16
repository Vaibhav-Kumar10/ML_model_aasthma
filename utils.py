import pandas as pd

def get_user_input():
    numerical_inputs = {
        "AQI": float(input("Enter AQI: ")),
        "PM2.5": float(input("Enter PM2.5 level: ")),
        "SO2 level": float(input("Enter SO2 level: ")),
        "NO2 level": float(input("Enter NO2 level: ")),
        "CO2 level": float(input("Enter CO2 level: ")),
        "Humidity": float(input("Enter Humidity (%): ")),
        "Temperature": float(input("Enter Temperature (°C): "))
    }

    categorical_options = {
        "Asthma Symptoms Frequency": ["Daily", "Frequently (Weekly)", "1-2 times a month"],
        "Triggers": ["Dust", "Temperature changes", "Air pollution (smoke, chemicals etc.)"],
        "Weather Sensitivity": ["Hot and humid weather", "Cold weather", "Windy and dry weather"],
        "Poor Air Quality Exposure": ["Yes, often", "Occasionally", "Never"],
        "Night Breathing Difficulty": ["Frequently", "Occasionally", "Rarely", "Never"]
    }

    categorical_inputs = {col: input(f"Choose {col} {options}: ") for col, options in categorical_options.items()}

    user_data = {**numerical_inputs, **categorical_inputs}
    return pd.DataFrame([user_data])
