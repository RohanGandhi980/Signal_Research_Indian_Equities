import os

DATA_BASE = "data_store"
RAW_DIR = os.path.join(DATA_BASE, "raw")
PROCESSED_DIR = os.path.join(DATA_BASE, "processed")
MODELS_DIR = os.path.join(DATA_BASE, "models_store")

import os

def raw_path_for(ticker: str) -> str:
    os.makedirs("data_store/raw", exist_ok=True)
    return f"data_store/raw/{ticker.replace('.', '_')}.csv"


for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

def raw_path_for(ticker: str) -> str:
    return os.path.join(RAW_DIR, f"{ticker}.csv")

def processed_path_for(ticker: str) -> str:
    return os.path.join(PROCESSED_DIR, f"{ticker}_processed.csv")

def model_path_for(ticker: str, model_name: str) -> str:
    return os.path.join(MODELS_DIR, f"{ticker}_{model_name}.pkl")

# --- backward-compatibility aliases so older imports still work ---
proc_path_for = processed_path_for
model_path = model_path_for

def proc_path_for(ticker: str) -> str:
    """Path to processed parquet file for a ticker."""
    return f"data/processed/{ticker.replace('.', '_')}_features.parquet"
