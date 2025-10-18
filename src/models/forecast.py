import os
import numpy as np
import pandas as pd
from src.models.sarima import SarimaForecaster
from src.models.lstm import LSTMForecaster

def predict_next_sarima(ticker: str):
    path = f"data/models/{ticker.replace('.', '_')}_sarima.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"SARIMA model not found for {ticker}. Train first.")
    model = SarimaForecaster.load(path)

    df = pd.read_parquet(f"data/processed/{ticker.replace('.', '_')}_features.parquet")
    y = df["log_return"] if "log_return" in df.columns else df["close"]

    pred_logret = model.predict_next(y)
    last_close = df["close"].iloc[-1]

    # Clamp extreme predictions
    pred_logret = np.clip(pred_logret, -0.3, 0.3)  # -30% to +30%
    predicted_close = float(last_close * np.exp(pred_logret))
    return round(predicted_close, 2)

def predict_next_lstm(ticker: str):
    path = f"data/models/{ticker.replace('.', '_')}_lstm.keras"
    if not os.path.exists(path):
        raise FileNotFoundError(f"LSTM model not found for {ticker}. Train first.")
    model = LSTMForecaster.load(path)

    df = pd.read_parquet(f"data/processed/{ticker.replace('.', '_')}_features.parquet")
    y = df["log_return"] if "log_return" in df.columns else df["close"]

    pred_logret = model.predict_next(y)
    pred_logret = np.clip(pred_logret, -0.3, 0.3)
    last_close = df["close"].iloc[-1]

    predicted_close = float(last_close * np.exp(pred_logret))
    return round(predicted_close, 2)
