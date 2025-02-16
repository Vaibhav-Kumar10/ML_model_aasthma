import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from preprocess import preprocess_data
import numpy as np
from tensorflow.keras import regularizers

# Load and prepare dataset
df = pd.read_csv("data/dataset.csv")
df.columns = df.columns.str.strip()

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Preprocess data
X_train, y_train, encoder, scaler = preprocess_data(train_df, train=True)
X_test, y_test, _, _ = preprocess_data(test_df, train=False, encoder=encoder, scaler=scaler)

def create_model(input_dim):
    inputs = keras.Input(shape=(input_dim,))

    x = keras.layers.Dense(512, kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('swish')(x)
    x = keras.layers.Dropout(0.3)(x)
    residual = x

    x = keras.layers.Dense(512, kernel_regularizer=regularizers.l2(0.01))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('swish')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Add()([x, residual])

    x = keras.layers.Dense(256, activation='swish', kernel_regularizer=regularizers.l2(0.01))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Dense(128, activation='swish', kernel_regularizer=regularizers.l2(0.01))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(64, activation='swish')(x)

    outputs = keras.layers.Dense(1, activation='linear')(x)

    return keras.Model(inputs, outputs)

print(f"Input feature dimension: {X_train.shape[1]}")

model = create_model(X_train.shape[1])

optimizer = keras.optimizers.AdamW(
    learning_rate=1e-3,
    weight_decay=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

model.compile(
    optimizer=optimizer,
    loss='huber',
    metrics=['mae', 'mse']
)

model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, min_delta=1e-4),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, save_weights_only=False)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=300,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nFinal Model Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")

model.save("model.keras")
with open("preprocessing.pkl", "wb") as f:
    pickle.dump((encoder, scaler), f)

print("\nModel training complete. Results saved.")
