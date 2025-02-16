import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from preprocess import preprocess_data
from tensorflow.keras import regularizers

# Load dataset
df = pd.read_csv("data/dataset.csv")
df.columns = df.columns.str.strip()

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Preprocess data (No Extra Features)
X_train, y_train, encoder, scaler = preprocess_data(train_df, train=True)
X_test, y_test, _, _ = preprocess_data(test_df, train=False, encoder=encoder, scaler=scaler)

# Define Model
def create_model(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(256, activation='swish', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(128, activation='swish')(x)
    outputs = keras.layers.Dense(1, activation='linear')(x)
    return keras.Model(inputs, outputs)

# Create and Compile Model
model = create_model(X_train.shape[1])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train Model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1)

# Save Model & Preprocessing Objects
model.save("model.keras")
with open("preprocessing.pkl", "wb") as f:
    pickle.dump((encoder, scaler), f)

print("\nModel training complete. Model and preprocessing objects saved.")
