from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
DATA_STORE = ROOT / "data_store"
RAW_DIR = DATA_STORE / "raw"
PROC_DIR = DATA_STORE / "processed"
MODELS_DIR = ROOT / "models_store"
LOG_FILE = ROOT / "signal_research.log"

for p in (DATA_STORE, RAW_DIR, PROC_DIR, MODELS_DIR):
    p.mkdir(parents=True, exist_ok=True)

DEFAULT_TICKER = "TCS.NS"        # NSE: *.NS ; BSE: *.BO
START_DATE = "2016-01-01"
END_DATE = None                  # today

TARGET_COL = "Close"             # we forecast next-day Close
FEATURE_PREFIXES = ("ma_", "std_", "mom_", "rsi_", "vol_")

# time splits
TEST_DAYS = 365
VAL_DAYS = 180

@dataclass
class LSTMConfig:
    window:int = 60
    units:int = 64
    dropout:float = 0.2
    lr:float = 1e-3
    epochs:int = 20
    batch:int = 64
