import os
import numpy as np
import pandas as pd
from src.data.ingest import ingest_data
from src.utils.logger import get_logger
from datetime import datetime

log = get_logger(__name__)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def build_features(ticker, start_date="2020-01-01", end_date=None, target_col="close"):
    """Safely build features and ensure valid date columns."""
    if end_date in [None, "string", ""]:
        end_date = datetime.today().strftime("%Y-%m-%d")

    log.info(f"[INFO] Building features for {ticker} up to {end_date}")

    df = ingest_data(ticker, start_date, end_date)
    if df.empty:
        raise ValueError(f"❌ No data fetched for {ticker}")

    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        df["date"] = pd.date_range(start=start_date, periods=len(df))

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.drop_duplicates(subset="date", inplace=True)

    # --- Feature engineering ---
    df["log_return"] = np.log(df[target_col] / df[target_col].shift(1))
    df["volatility_20"] = df["log_return"].rolling(20).std()
    df["ma_5"] = df[target_col].rolling(5).mean()
    df["ma_20"] = df[target_col].rolling(20).mean()
    df["rsi_14"] = compute_rsi(df[target_col])

    df.dropna(inplace=True)

    os.makedirs("data/processed", exist_ok=True)
    output_path = f"data/processed/{ticker.replace('.', '_')}_features.parquet"
    df.to_parquet(output_path, index=False)
    log.info(f"[INFO] ✅ Features saved to {output_path}")
    return df
