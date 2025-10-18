import pandas as pd
from src.utils.paths import raw_path_for, processed_path_for
from src.utils.logger import get_logger

log = get_logger(__name__)

def make_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute returns, moving averages, and volatility features."""
    df["Return"] = df["AdjClose"].pct_change()
    df["MA5"] = df["AdjClose"].rolling(5).mean()
    df["MA20"] = df["AdjClose"].rolling(20).mean()
    df["Volatility"] = df["Return"].rolling(20).std()
    return df.dropna()

def save_processed(df: pd.DataFrame, ticker: str):
    """Persist processed features to data_store/processed."""
    path = processed_path_for(ticker)
    df.to_csv(path, index=True)
    log.info(f"💾 Processed data saved → {path}")
