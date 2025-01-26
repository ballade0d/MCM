import numpy as np
import pandas as pd
from keras import Input, Model
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Bidirectional, LSTM, Dense, Multiply
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

country = 'United States'

# Data reading and initial processing
medal_counts = pd.read_csv('data/summerOly_medal_counts.csv')
hosts = pd.read_csv('data/summerOly_hosts.csv')

medal_counts = medal_counts[medal_counts['NOC'] == country]
years = medal_counts['Year'].values

X_extra = []
for year in years:
    is_host = hosts[(hosts['Year'] == year) & (hosts['Host'] == country)].any().any()
    X_extra.append(1 if is_host else 0)
X_extra = np.array(X_extra).reshape(-1, 1)
X_raw = medal_counts['Total'].values.reshape(-1, 1)
X_raw = np.hstack((X_raw, X_extra))

# Data scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(X_raw)


# Function to create lagged data
def create_lagged_data(data, lag):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def evaluate_model_with_lag(lag):
    # Create lagged data
    X, y = create_lagged_data(scaled_values, lag)

    # Split into training and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Extract host variable from X_train and X_test
    host_train = X_train[:, -1, 1].reshape(-1, 1)
    host_test = X_test[:, -1, 1].reshape(-1, 1)

    # Build the model
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    host_input = Input(shape=(1, 1))
    attention = Dense(1, activation='sigmoid')(host_input)
    weighted_input = Multiply()([input_layer, attention])
    lstm = Bidirectional(LSTM(64, return_sequences=False))(weighted_input)
    lstm = Dense(32)(lstm)
    output_layer = Dense(1)(lstm)

    model = Model(inputs=[input_layer, host_input], outputs=[output_layer])
    model.compile(optimizer='adam', loss='mae')

    # Train the model
    history = model.fit(
        [X_train, host_train], y_train,
        epochs=100,
        batch_size=32,
        verbose=0,
        validation_data=([X_test, host_test], y_test),
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.8, patience=10)
        ],
    )

    # Get the best validation loss
    val_loss = min(history.history['val_loss'])
    return val_loss


# Loop over lag values and collect performance metrics
lags = range(1, 11)
performance_metrics = []

for lag in lags:
    val_loss = evaluate_model_with_lag(lag)
    performance_metrics.append(val_loss)
    print(f'Lag: {lag}, Validation Loss: {val_loss:.4f}')

# Find the lag with minimum validation loss
best_lag_index = np.argmin(performance_metrics)
best_lag = lags[best_lag_index]
best_val_loss = performance_metrics[best_lag_index]
print(f'Best lag is {best_lag} with validation loss {best_val_loss:.4f}')