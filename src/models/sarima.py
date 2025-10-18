import os
import pickle
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SarimaConfig:
    order: tuple = (1, 1, 1)
    seasonal_order: tuple = (0, 1, 1, 12)


class SarimaForecaster:
    """
    Wrapper class for training, forecasting, saving, and loading SARIMA models.
    """

    def __init__(self, cfg: SarimaConfig = SarimaConfig()):
        self.cfg = cfg
        self.model = None
        self.results = None

    def fit(self, y: pd.Series):
        """Train SARIMA model on a time series."""
        log.info("📊 Fitting SARIMA model...")
        self.model = sm.tsa.statespace.SARIMAX(
            y,
            order=self.cfg.order,
            seasonal_order=self.cfg.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.results = self.model.fit(disp=False)
        log.info("✅ SARIMA training complete.")
        return self

    def predict_next(self, y: pd.Series) -> float:
        """Forecast the next value."""
        if self.results is None:
            raise ValueError("Model not trained. Call .fit() first.")
        forecast = self.results.get_forecast(steps=1)
        yhat = forecast.predicted_mean.iloc[0]
        log.info(f"🔮 Next forecast: {yhat:.2f}")
        return float(yhat)

    def save(self, path: str):
        """Save model using pickle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.results, f)
        log.info(f"💾 SARIMA model saved → {path}")

    @staticmethod
    def load(path: str):
        """Load model from disk."""
        with open(path, "rb") as f:
            results = pickle.load(f)
        obj = SarimaForecaster()
        obj.results = results
        log.info(f"📂 SARIMA model loaded → {path}")
        return obj


import os
import pickle
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.utils.logger import get_logger

log = get_logger(__name__)

def train_sarima_model(df: pd.DataFrame, ticker: str):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("❌ Neither Date column nor DatetimeIndex found in dataframe.")

    y = df["log_return"] if "log_return" in df.columns else df["close"]

    model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    result = model.fit(disp=False)

    os.makedirs("data/models", exist_ok=True)
    path = f"data/models/{ticker.replace('.', '_')}_sarima.pkl"
    with open(path, "wb") as f:
        pickle.dump(result, f)

    log.info(f"✅ SARIMA model saved → {path}")
    return result



