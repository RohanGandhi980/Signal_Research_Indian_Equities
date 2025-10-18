from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_STORE = ROOT / "data_store"
RAW_DIR = DATA_STORE / "raw"
PROC_DIR = DATA_STORE / "processed"
MODELS_DIR = ROOT / "models_store"
LOG_FILE = ROOT / "signal_research.log"

for p in (DATA_STORE, RAW_DIR, PROC_DIR, MODELS_DIR):
    p.mkdir(parents=True, exist_ok=True)

def load_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)
