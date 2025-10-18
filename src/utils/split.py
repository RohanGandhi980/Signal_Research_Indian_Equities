import pandas as pd
from typing import Tuple
from src.utils.logger import get_logger

log = get_logger(__name__)

def time_splits(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split a time-series DataFrame into train and test sets.
    Keeps the first `train_ratio` portion for training and the rest for testing.
    """
    n = len(df)
    split_idx = int(train_ratio * n)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    log.info(f"🕒 Split complete: Train={len(train_df)} rows, Test={len(test_df)} rows")
    return train_df, test_df
