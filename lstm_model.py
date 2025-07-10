# lstm_model.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


def prepare_lstm_data(series, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # 3D input for LSTM
    return X, y, scaler


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_lstm_model(df_close, epochs=20, batch_size=32):
    X, y, scaler = prepare_lstm_data(df_close)
    model = build_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, scaler, X[-1].reshape(1, -1, 1)  # Return model, scaler, last known sequence


def predict_next_days(model, last_sequence, scaler, n_days=5):
    future_preds = []
    current_seq = last_sequence

    for _ in range(n_days):
        next_pred = model.predict(current_seq, verbose=0)[0][0]
        future_preds.append(next_pred)
        next_seq = np.append(current_seq[0, 1:, 0], next_pred)
        current_seq = next_seq.reshape(1, -1, 1)

    predicted_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    return predicted_prices
