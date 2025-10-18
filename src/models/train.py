import pandas as pd
from src.utils.logger import get_logger
from src.models.sarima import train_sarima_model
from src.models.lstm import train_lstm_model

log=get_logger(__name__)

def train_sarima(ticker:str):
    log.info(f"📈 Training SARIMA for {ticker}")
    path=f"data/processed/{ticker.replace('.', '_')}_features.parquet"
    df=pd.read_parquet(path)
    return train_sarima_model(df,ticker)

def train_lstm(ticker:str):
    log.info(f"🧠 Training LSTM for {ticker}")
    path=f"data/processed/{ticker.replace('.', '_')}_features.parquet"
    df=pd.read_parquet(path)
    return train_lstm_model(df,ticker)
