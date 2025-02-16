import pickle
import tensorflow as tf
from preprocess import preprocess_data
from utils import get_user_input

# Load trained model and preprocessing objects
model = tf.keras.models.load_model("model.keras")
with open("preprocessing.pkl", "rb") as f:
    encoder, scaler = pickle.load(f)

# Get user input
user_df = get_user_input()

# Preprocess user input
X_processed, _, _, _ = preprocess_data(user_df, train=False, encoder=encoder, scaler=scaler)

# Make prediction
prediction = model.predict(X_processed)[0][0]
print(f"Predicted Asthma Risk Factor: {prediction:.2f}")
