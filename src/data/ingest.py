import os
import pandas as pd
from datetime import datetime
from yahooquery import Ticker
from src.utils.paths import raw_path_for
from src.utils.logger import get_logger

log = get_logger(__name__)

def fetch_yahoo(ticker: str, start_date: str = "2020-01-01", end_date: str | None = None) -> pd.DataFrame:
    """Fetch OHLCV data up to today's date if end_date not provided."""
    if not end_date:
        end_date = datetime.today().strftime("%Y-%m-%d")

    if not ticker.endswith((".NS", ".BO")):
        ticker = f"{ticker}.NS"

    log.info(f"📥 Fetching {ticker} from {start_date} to {end_date}")
    os.makedirs("data_store/raw", exist_ok=True)

    t = Ticker(ticker)
    hist = t.history(start=start_date, end=end_date, adj_ohlc=True)
    if hist.empty:
        log.warning(f"⚠️ No data found for {ticker}")
        return pd.DataFrame()

    hist = hist.reset_index().rename(columns=str.title)
    hist = hist[hist["Symbol"] == ticker].drop(columns=["Symbol"], errors="ignore")
    hist = hist.rename(columns={"Adjclose": "AdjClose"}).set_index("Date")

    out = raw_path_for(ticker)
    hist.to_csv(out)
    log.info(f"✅ Saved raw data → {out}")
    return hist


def ingest_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical OHLCV data using YahooQuery."""
    log.info(f"[INFO] Downloading data for {ticker} from {start_date} to {end_date}")

    try:
        ticker_obj = Ticker(ticker)
        hist = ticker_obj.history(start=start_date, end=end_date)
        if hist.empty:
            raise ValueError("No data returned from YahooQuery.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to fetch {ticker}: {e}")

    # Reset and clean
    if isinstance(hist.index, pd.MultiIndex):
        hist.reset_index(inplace=True)
    if "date" not in hist.columns:
        hist.rename(columns={"asOfDate": "date"}, inplace=True)
    hist["date"] = pd.to_datetime(hist["date"])
    hist.sort_values("date", inplace=True)
    hist = hist[["date", "open", "high", "low", "close", "volume"]]

    # Fill missing closes if any
    hist["close"].ffill(inplace=True)
    hist.dropna(subset=["close"], inplace=True)

    return hist